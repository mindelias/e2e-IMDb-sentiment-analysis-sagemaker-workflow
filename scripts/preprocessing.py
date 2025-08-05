import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()

import re
from bs4 import BeautifulSoup



def review_to_words(review):
    # Import inside function to ensure packages are installed first
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    """Convert a raw review string into a cleaned word list"""
    # 1. Remove HTML tags
    text = BeautifulSoup(review, "html.parser").get_text()
    
    # 2. Keep only letters/numbers and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # 3. Split into individual words
    words = text.split()
    
    # 4. Remove common stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    
    # 5. Reduce words to their root form
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)  