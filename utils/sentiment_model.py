import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModel

class SentimentModels:
    def __init__(self):
        self.baseline_model = LogisticRegression(max_iter=1000)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        # Advanced Model components (DistilBERT + Classifier)
        self.advanced_classifier = LogisticRegression(max_iter=1000)
        self.tokenizer = None
        self.bert_model = None
        
        self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        self.reverse_mapping = {0: "negative", 1: "neutral", 2: "positive"}

    def _init_bert(self):
        """Loads DistilBERT for feature extraction if not already loaded."""
        if self.tokenizer is None or self.bert_model is None:
            print("Loading DistilBERT model (this may take a minute on first run)...")
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
            self.bert_model.eval()

    def _get_bert_embeddings(self, texts):
        """Extracts BERT embeddings for a list of texts."""
        self._init_bert()
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output = self.bert_model(**encoded)
            # Use the mean pooling of the last hidden state as sentence embedding
            batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)

    def prepare_data(self, df):
        """Prepares labels and text arrays from dataframe."""
        texts = df['processed_text'].tolist()
        
        # Rating to label logic
        def convert_rating(r):
            if r >= 4: return "positive"
            elif r == 3: return "neutral"
            else: return "negative"
            
        labels = df['rating'].apply(convert_rating).map(self.label_mapping).tolist()
        return texts, np.array(labels)

    def train_evaluate(self, train_df, test_df):
        """Trains and compares both baseline and advanced models."""
        X_train_text, y_train = self.prepare_data(train_df)
        X_test_text, y_test = self.prepare_data(test_df)
        
        print("\n--- Training Baseline Model (TF-IDF + Logistic Regression) ---")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        
        self.baseline_model.fit(X_train_tfidf, y_train)
        base_preds = self.baseline_model.predict(X_test_tfidf)
        base_metrics = self._calculate_metrics(y_test, base_preds)
        
        print("\n--- Training Advanced Model (DistilBERT + Extractor) ---")
        print("Extracting features from training set...")
        X_train_bert = self._get_bert_embeddings(X_train_text)
        print("Extracting features from testing set...")
        X_test_bert = self._get_bert_embeddings(X_test_text)
        
        self.advanced_classifier.fit(X_train_bert, y_train)
        adv_preds = self.advanced_classifier.predict(X_test_bert)
        adv_metrics = self._calculate_metrics(y_test, adv_preds)
        
        self._print_comparison(base_metrics, adv_metrics)
        self.save_models()

    def _calculate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return {"Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1, "Confusion Matrix": cm}

    def _print_comparison(self, base_m, adv_m):
        print("\n================ MODEL COMPARISON ================")
        print(f"{'Metric':<15} | {'Baseline (TF-IDF)':<20} | {'Advanced (BERT)':<20}")
        print("-" * 60)
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            print(f"{metric:<15} | {base_m[metric]:<20.4f} | {adv_m[metric]:<20.4f}")
        
        print("\nAdvanced Model (BERT) Confusion Matrix:")
        print(adv_m["Confusion Matrix"])
        print("==================================================\n")

    def save_models(self):
        """Saves models to the models/ directory."""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        with open('models/baseline_model.pkl', 'wb') as f:
            pickle.dump((self.vectorizer, self.baseline_model), f)
            
        with open('models/advanced_model.pkl', 'wb') as f:
            pickle.dump(self.advanced_classifier, f)
        print("Models successfully saved to 'models/' directory.")

    def load_models(self):
        """Loads models from the models/ directory."""
        with open('models/baseline_model.pkl', 'rb') as f:
            self.vectorizer, self.baseline_model = pickle.load(f)
            
        with open('models/advanced_model.pkl', 'rb') as f:
            self.advanced_classifier = pickle.load(f)

    def predict_sentiment(self, text, use_advanced=True):
        """Predicts sentiment for a preprocessed text."""
        if use_advanced:
            self._init_bert()
            emb = self._get_bert_embeddings([text])
            pred_idx = self.advanced_classifier.predict(emb)[0]
            # Probabilities
            probs = self.advanced_classifier.predict_proba(emb)[0]
        else:
            vec = self.vectorizer.transform([text])
            pred_idx = self.baseline_model.predict(vec)[0]
            probs = self.baseline_model.predict_proba(vec)[0]
            
        label = self.reverse_mapping[pred_idx]
        score = probs[pred_idx] # Confidence score of the predicted class
        
        # We need a continuous sentiment score for ranking: 
        # Map to -1 (negative), 0 (neutral), 1 (positive) weighted by confidence
        if label == "positive":
            continuous_score = score
        elif label == "negative":
            continuous_score = -score
        else:
            continuous_score = 0.0
            
        # Normalize continuous score between 0 and 1 for easier ranking computation later
        normalized_score = (continuous_score + 1.0) / 2.0
        
        return label, normalized_score
