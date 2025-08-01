import os
import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scripts.preprocessing import review_to_words  # Custom cleaning functions

def load_dataset(path):
    """
    Load raw reviews from directory structure
    - path: Directory with 'pos' and 'neg' subfolders
    Returns: DataFrame with columns ['review', 'sentiment']
    """
    reviews = []
    # 1 = Positive, 0 = Negative
    for label, sentiment in [('pos', 1), ('neg', 0)]:
        folder = os.path.join(path, label)
        # Process each review file
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                reviews.append([f.read(), sentiment])
    return pd.DataFrame(reviews, columns=['review', 'sentiment'])

if __name__ == "__main__":
    # SageMaker passes these paths automatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input/data')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()

    # 1. LOAD RAW DATA ---------------------------------------------------------
    # Original dataset structure:
    #   input-data/
    #     train/pos/*.txt  - 12,500 positive reviews
    #     train/neg/*.txt  - 12,500 negative reviews
    #     test/pos/*.txt   - 12,500 positive reviews
    #     test/neg/*.txt   - 12,500 negative reviews
    
    train_df = load_dataset(os.path.join(args.input_data, 'train'))  # 25,000 reviews
    test_df = load_dataset(os.path.join(args.input_data, 'test'))    # 25,000 reviews

    # 2. CLEAN TEXT ------------------------------------------------------------
    # Apply the same cleaning to all reviews
    train_df['cleaned_review'] = train_df['review'].apply(review_to_words)
    test_df['cleaned_review'] = test_df['review'].apply(review_to_words)

    # 3. SPLIT DATA PROPERLY --------------------------------------------------
    # Best practice:
    #   - Train: Model training (80% of total training data)
    #   - Validation: Hyperparameter tuning (20% of total training data)
    #   - Test: Final evaluation (100% of original test set)
    
    # Split ORIGINAL TRAINING DATA into train/validation
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.2,  # 20% for validation
        random_state=42,
        stratify=train_df['sentiment']  # Maintain class balance
    )
    # test_df remains untouched for final evaluation

    # 4. VECTORIZE TEXT -------------------------------------------------------
    # Critical: Fit ONLY on training data to avoid data leakage
    vectorizer = CountVectorizer(
        max_features=5000,  # Use top 5000 words
        binary=True         # Use 1/0 (present/absent) instead of counts
    )
    
    # Learn vocabulary ONLY from training split
    X_train = vectorizer.fit_transform(train_data['cleaned_review'])
    # Apply SAME transformation to other datasets
    X_val = vectorizer.transform(val_data['cleaned_review'])
    X_test = vectorizer.transform(test_df['cleaned_review'])

    # 5. CREATE FINAL DATASETS ------------------------------------------------
    # Combine labels with features
    train_processed = pd.concat([
        pd.DataFrame(train_data['sentiment'].values, columns=['sentiment']),
        pd.DataFrame(X_train.toarray())
    ], axis=1)
    
    val_processed = pd.concat([
        pd.DataFrame(val_data['sentiment'].values, columns=['sentiment']),
        pd.DataFrame(X_val.toarray())
    ], axis=1)
    
    test_processed = pd.concat([
        pd.DataFrame(test_df['sentiment'].values, columns=['sentiment']),
        pd.DataFrame(X_test.toarray())
    ], axis=1)

    # 6. SAVE PROCESSED DATA --------------------------------------------------
    # SageMaker requires specific directory structure
    os.makedirs(f'{args.output_data}/train', exist_ok=True)
    os.makedirs(f'{args.output_data}/validation', exist_ok=True)
    os.makedirs(f'{args.output_data}/test', exist_ok=True)
    os.makedirs(f'{args.output_data}/vectorizer', exist_ok=True)
    
    # Save without headers/index (XGBoost requirement)
    train_processed.to_csv(
        f'{args.output_data}/train/train.csv', 
        index=False, 
        header=False
    )
    val_processed.to_csv(
        f'{args.output_data}/validation/validation.csv', 
        index=False, 
        header=False
    )
    test_processed.to_csv(
        f'{args.output_data}/test/test.csv', 
        index=False, 
        header=False
    )
    
    # MUST SAVE VECTORIZER FOR CONSISTENT INFERENCE
    joblib.dump(vectorizer, f'{args.output_data}/vectorizer/vectorizer.joblib')
    print(f"✅ Processing complete. Vocabulary size: {len(vectorizer.vocabulary_)}")














