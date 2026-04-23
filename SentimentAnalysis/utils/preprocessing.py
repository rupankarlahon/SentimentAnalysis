import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    """Class to handle text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def _download_nltk_resources(self):
        """Ensure required NLTK resources are available."""
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
        for res in resources:
            try:
                # NLTK find expects specific subpaths, relying on download logic instead
                nltk.download(res, quiet=True)
            except Exception as e:
                pass

    def process(self, text):
        """
        Applies lowercasing, punctuation removal, tokenization, 
        stopword removal, and lemmatization.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercasing
        text = text.lower()
        
        # 2. Removing punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4 & 5. Stopword removal & Lemmatization
        processed_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and word.strip()
        ]
        
        return " ".join(processed_tokens)
