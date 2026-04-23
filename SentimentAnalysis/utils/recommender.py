import pandas as pd
from utils.food_extractor import FoodExtractor
from utils.preprocessing import Preprocessor
from utils.sentiment_model import SentimentModels

class Recommender:
    def __init__(self, data_path="data/majitar_reviews.csv"):
        self.df = pd.read_csv(data_path)
        self.food_extractor = FoodExtractor()
        
        self.preprocessor = Preprocessor()
        self.models = SentimentModels()
        self.models.load_models()

    def recommend(self, food_item, use_advanced=True):
        """
        1. Filters reviews containing food_item
        2. Computes sentiment for matching reviews
        3. Groups by restaurant and computes final weighted score
        """
        food_item = food_item.lower().strip()
        matched_reviews = []
        
        for _, row in self.df.iterrows():
            review_text = str(row['review_text'])
            restaurant_name = row['restaurant_name']
            
            # Simple keyword filtering
            if food_item in review_text.lower():
                matched_reviews.append({
                    "restaurant": restaurant_name,
                    "review": review_text
                })
                
        if not matched_reviews:
            return [] # No restaurants found for this item
            
        restaurant_stats = {}
        for item in matched_reviews:
            rest = item['restaurant']
            rev_raw = item['review']
            
            # Predict Sentiment on the fly
            rev_processed = self.preprocessor.process(rev_raw)
            label, norm_score = self.models.predict_sentiment(rev_processed, use_advanced=use_advanced)
            
            if rest not in restaurant_stats:
                restaurant_stats[rest] = {"total_sentiment": 0.0, "count": 0}
                
            restaurant_stats[rest]["total_sentiment"] += norm_score
            restaurant_stats[rest]["count"] += 1
            
        # Rank the restaurants
        results = []
        max_count = max([stats["count"] for stats in restaurant_stats.values()])
        
        for rest, stats in restaurant_stats.items():
            avg_sentiment = stats["total_sentiment"] / stats["count"]
            normalized_count = stats["count"] / max_count if max_count > 0 else 0
            
            # Equation required by the user prompt
            final_score = (0.7 * avg_sentiment) + (0.3 * normalized_count)
            
            results.append({
                "restaurant": rest,
                "score": final_score,
                "avg_sentiment": avg_sentiment,
                "matching_reviews": stats["count"]
            })
            
        # Sort by best score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
