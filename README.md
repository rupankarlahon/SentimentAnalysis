#  Majitar Restaurant Sentiment & Recommendation System

Welcome to the **Sentiment-Based Food Recommendation System**! Have you ever craved a specific dish (like Momos, Pizza, or Coffee) but didn't know which restaurant in Majitar cooked it best? 

Instead of relying on basic 5-star ratings or scrolling through hundreds of reviews to see if the food was *actually* good, this tool uses powerful Artificial Intelligence (Natural Language Processing) to scrape real Google Reviews, analyze the sentiment of what people are actually saying, and rank the absolute best places to eat in your terminal.

---

##  What Does It Do?

1. **Scrapes Real Data:** Uses `SerpApi` to seamlessly fetch the latest authentic Google reviews for restaurants around Majitar, Sikkim.
2. **Deep Text Understanding:** Cleans the textual data using `NLTK` (lowercasing, punctuation stripping, and WordNet lemmatization).
3. **Advanced AI Sentiment Analysis:** Employs Hugging Face's **DistilBERT** embeddings paired with a Logistic Regression classifier to achieve **93% accuracy** in understanding whether a review is positive or negative.
4. **Smart Search CLI:** A beautiful, responsive terminal interface that asks you what you want to eat, filters the reviews, and computes a balanced ranking score (`70% Sentiment Quality` + `30% Review Volume`).

---

##  Setup & Installation

### 1. Install the Requirements
Make sure you have Python installed, then install the necessary AI and data packages:
```bash
pip install -r requirements.txt
```

### 2. Fetch the Dataset (Optional but Recommended)
If you want to pull fresh, live data from Google Maps, you will need a free API key from [SerpApi](https://serpapi.com/). 
Set your API key in your terminal and run the scraper:

**Windows (PowerShell):**
```powershell
$env:SERPAPI_KEY="your_serpapi_key_here"
python data/fetch_reviews.py
```
*(This will save hundreds of reviews into `data/majitar_reviews.csv`!)*

### 3. Train the AI Models
Before searching, we need to teach our models how to read the reviews! Run the training script:
```bash
python train_models.py
```
This script will process the text, train a Baseline TF-IDF model and an Advanced DistilBERT model, compare their accuracy, and safely export their "brains" into the `models/` folder.

---

##  How to Use the Recommender

Once the setup is done, launch the interactive terminal tool:
```bash
python main.py
```

It will greet you and ask what you're craving. Type in foods like `momo`, `coffee`, or `pizza`!
```text
Enter food item (or 'quit' to exit): momo

Food searched: momo

1. Tashi Shaa-Khang Chinese restaurant   Score: 1.00
2. SK01 , FOOD•TOUR•CULTURE              Score: 0.99
```

---

##  Project Structure

```text
/project
│   main.py               #  The interactive CLI script
│   train_models.py       #  Trains and evaluates the Baseline vs Advanced models
│   requirements.txt      #  Python library dependencies
│   README.md             #  You are here!
│
├── data/
│     fetch_reviews.py        # Connects to SerpApi to scrape local restaurants
│     majitar_reviews.csv     # The raw scraped reviews
│     processed_reviews.csv   # The NLTK cleaned reviews
│
├── models/
│     baseline_model.pkl      # Saved TF-IDF weights
│     advanced_model.pkl      # Saved BERT + LR weights
│
└── utils/
      preprocessing.py    # Handles Stopwords, Lemmatization, Tokenization
      sentiment_model.py  # Contains the actual ML Model pipelines
      food_extractor.py   # Uses Regex patterns to identify food requests
      recommender.py      # The mathematical engine that ranks the restaurants
```

Enjoy finding the best food in Majitar! 
