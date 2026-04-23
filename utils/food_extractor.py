import re

class FoodExtractor:
    def __init__(self, custom_keywords=None):
        if custom_keywords:
            self.food_keywords = [f.lower() for f in custom_keywords]
        else:
            self.food_keywords = [
                "momos", "pizza", "burger", "thukpa", 
                "chowmein", "fried rice", "coffee", "biryani",
                "roll", "noodles", "momo"
            ]

    def extract(self, text):
        """Extracts food keywords from text using a predefined list."""
        if not isinstance(text, str):
            return []
            
        found_foods = set()
        text_lower = text.lower()
        
        for food in self.food_keywords:
            # Using regex word boundary to match the food phrase
            if re.search(rf'\b{re.escape(food)}\b', text_lower):
                found_foods.add(food)
                
        return list(found_foods)
