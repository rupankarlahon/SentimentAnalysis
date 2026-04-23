import pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessor
from utils.sentiment_model import SentimentModels

def train_pipeline():
    print("Loading dataset...")
    try:
        df = pd.read_csv("data/majitar_reviews.csv")
    except FileNotFoundError:
        print("Error: data/majitar_reviews.csv not found. Please run fetch_reviews.py first.")
        return

    print("Preprocessing text (this may take a moment)...")
    preprocessor = Preprocessor()
    # Apply preprocessing (filling NaNs with empty string)
    df['processed_text'] = df['review_text'].fillna("").apply(preprocessor.process)
    
    # Save the processed data
    df.to_csv("data/processed_reviews.csv", index=False)
    
    print("Splitting data into train/test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples.")
    models = SentimentModels()
    models.train_evaluate(train_df, test_df)

if __name__ == "__main__":
    train_pipeline()
