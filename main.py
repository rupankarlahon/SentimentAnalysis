import sys
from utils.recommender import Recommender

def main():
    print("===============================================================")
    print(" Sentiment-Based Food Recommendation System (Majitar Edition)  ")
    print("===============================================================")
    
    print("\nLoading models and dataset... Please wait.")
    try:
        recommender = Recommender()
    except Exception as e:
        print(f"\nError: Failed to initialize Recommender ({e})")
        print("Please ensure you've run 'python data/fetch_reviews.py' and 'python train_models.py'.")
        sys.exit(1)
        
    print("System Ready!\n")
    
    while True:
        try:
            query = input("Enter food item (or 'quit' to exit): ").strip()
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting. Enjoy your meal!")
                break
                
            print(f"\nFood searched: {query}\n")
            
            # Predict and Rank (Using the Best BERT model by default)
            results = recommender.recommend(query, use_advanced=True)
            
            if not results:
                print(f"No Google reviews found mentioning '{query}'.")
                print("Try another item like 'pizza', 'burger', or 'coffee'.\n")
                continue
                
            # Display Top 5
            top_5 = results[:5]
            for idx, res in enumerate(top_5, 1):
                # Padding restaurant name for the clean aligned format requested
                rest_name = f"{res['restaurant']:<20}"
                score = f"{res['score']:.2f}"
                print(f"{idx}. {rest_name} Score: {score}")
                
            print("\n" + "-" * 55 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting. Enjoy your meal!")
            break

if __name__ == "__main__":
    main()
