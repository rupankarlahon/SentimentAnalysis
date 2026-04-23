import os
import time
import requests
import pandas as pd

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "066724485c038db6841557a81cdbf87d38f45a2ac6624c09e999739995789c49")

def get_majitar_restaurants():
    """Fetches a list of restaurants in Majitar to get their data_id."""
    print("Searching for restaurants in Majitar...")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_maps",
        "q": "restaurants in Majitar, Sikkim",
        "type": "search",
        "api_key": SERPAPI_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    restaurants = []
    if "local_results" in data:
        for place in data["local_results"]:
            restaurants.append({
                "name": place.get("title"),
                "data_id": place.get("data_id"),
                "rating": place.get("rating")
            })
    return restaurants

def get_restaurant_reviews(data_id, num_pages=2):
    """Fetches reviews for a specific restaurant using its data_id."""
    reviews = []
    next_page_token = None
    
    for page in range(num_pages):
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_maps_reviews",
            "data_id": data_id,
            "api_key": SERPAPI_KEY
        }
        
        if next_page_token:
            params["next_page_token"] = next_page_token
            
        res = requests.get(url, params=params)
        data = res.json()
        
        if "reviews" in data:
            for review in data["reviews"]:
                # Only keep reviews with text
                if review.get("snippet"):
                    reviews.append({
                        "review_text": review.get("snippet"),
                        "rating": review.get("rating")
                    })
                    
        # Check for pagination
        if "serpapi_pagination" in data and "next_page_token" in data["serpapi_pagination"]:
            next_page_token = data["serpapi_pagination"]["next_page_token"]
        else:
            break
            
        time.sleep(1) # Be nice to the API
        
    return reviews

def main():
    if SERPAPI_KEY == "066724485c038db6841557a81cdbf87d38f45a2ac6624c09e999739995789c49":
        print("ERROR: Please set your SerpApi key in the script or via SERPAPI_KEY environment variable.")
        return

    restaurants = get_majitar_restaurants()
    if not restaurants:
        print("No restaurants found. Please check your API key and query.")
        return
        
    print(f"Found {len(restaurants)} restaurants. Fetching reviews...")
    
    all_reviews = []
    for rest in restaurants:
        if not rest["data_id"]:
            continue
            
        print(f"  -> Fetching reviews for {rest['name']}...")
        rest_reviews = get_restaurant_reviews(rest["data_id"], num_pages=5) # 5 pages = ~50 reviews per restaurant
        
        for r in rest_reviews:
            r["restaurant_name"] = rest["name"]
            all_reviews.append(r)
            
    # Save to DataFrame and export to CSV
    df = pd.DataFrame(all_reviews)
    
    # Reorder columns
    if not df.empty:
        df = df[["restaurant_name", "review_text", "rating"]]
        output_file = "data/majitar_reviews.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} reviews to {output_file}!")
    else:
        print("No reviews were found.")

if __name__ == "__main__":
    main()
