import os
import argparse
import pandas as pd
import joblib
import re
import sys
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Install required packages at runtime
def install_packages():
    packages = ['beautifulsoup4==4.12.3', 'nltk==3.8.1']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])



# Define the text cleaning function directly in this script
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


if __name__ == "__main__":
    # Install packages before anything else
    print("🔧 Installing required packages...")
    install_packages()

    # Download NLTK data
    print("📥 Downloading NLTK data...")
    import nltk
    nltk.download('stopwords')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input/data')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()


    

    # 1. LOAD PREPROCESSED CSV ---------------------------------------------------
    print("📥 Loading CSV files...")
    train_df = pd.read_csv(os.path.join(args.input_data, 'train.csv'))  # ['review', 'sentiment']
    test_df = pd.read_csv(os.path.join(args.input_data, 'test.csv'))    # ['review', 'sentiment']

    # 2. CLEAN TEXT -------------------------------------------------------------
    print("🧹 Cleaning text...")
    train_df['cleaned_review'] = train_df['review'].apply(review_to_words)
    test_df['cleaned_review'] = test_df['review'].apply(review_to_words)

    # 3. SPLIT INTO TRAIN/VALIDATION -------------------------------------------
    print("✂️ Splitting train into train/validation...")
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['sentiment']
    )

    # 4. VECTORIZE --------------------------------------------------------------
    print("🧠 Fitting vectorizer...")
    vectorizer = CountVectorizer(max_features=5000, binary=True)
    X_train = vectorizer.fit_transform(train_data['cleaned_review'])
    X_val = vectorizer.transform(val_data['cleaned_review'])
    X_test = vectorizer.transform(test_df['cleaned_review'])

    # 5. PREPARE FINAL DATASETS -------------------------------------------------
    print("📦 Preparing final datasets...")
    def combine(X, y):
        return pd.concat([
            pd.DataFrame(y.values, columns=['sentiment']),
            pd.DataFrame(X.toarray())
        ], axis=1)

    train_processed = combine(X_train, train_data['sentiment'])
    val_processed = combine(X_val, val_data['sentiment'])
    test_processed = combine(X_test, test_df['sentiment'])

    # 6. SAVE OUTPUTS -----------------------------------------------------------
    print("💾 Saving outputs...")
    os.makedirs(f'{args.output_data}/train', exist_ok=True)
    os.makedirs(f'{args.output_data}/validation', exist_ok=True)
    os.makedirs(f'{args.output_data}/test', exist_ok=True)
    os.makedirs(f'{args.output_data}/vectorizer', exist_ok=True)

    train_processed.to_csv(f'{args.output_data}/train/train.csv', index=False, header=False)
    val_processed.to_csv(f'{args.output_data}/validation/validation.csv', index=False, header=False)
    test_processed.to_csv(f'{args.output_data}/test/test.csv', index=False, header=False)

    joblib.dump(vectorizer, f'{args.output_data}/vectorizer/vectorizer.joblib')

    print(f"✅ Done! Vectorizer vocab size: {len(vectorizer.vocabulary_)}")
