# =================================================================
# SAGEMAKER END-TO-END SENTIMENT ANALYSIS PIPELINE
# =================================================================

import os
import re
import glob
import pickle
import joblib
import pandas as pd
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# =================================================================
# 1. DATA ACQUISITION & PREPROCESSING
# =================================================================

def download_imdb_data():
    """Fetches IMDb dataset if not already available locally"""
    # IMDb dataset is publicly available - we'll download it once
    # This contains 50,000 movie reviews with sentiment labels
    data_dir = '../data/aclImdb'
    if not os.path.exists(data_dir):
        print("📥 Downloading IMDb dataset...")
        # Simple way to get the dataset - in production we'd use more robust methods
        os.system('wget -P ../data/ https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
        os.system('tar -xvzf ../data/aclImdb_v1.tar.gz -C ../data/')
    return data_dir

def review_to_words(raw_review):
    """Cleans raw text into machine-readable words"""
    # 1. Remove HTML tags - reviews often contain HTML formatting
    text = BeautifulSoup(raw_review, "html.parser").get_text()
    
    # 2. Keep only letters/numbers - removes punctuation/special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # 3. Split into individual words
    words = text.split()
    
    # 4. Remove stopwords - common words like "the", "and", "is" that don't add meaning
    words = [w for w in words if w not in stopwords.words("english")]
    
    # 5. Stemming - reduce words to root form (e.g., "running" → "run")
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    
    return words

# =================================================================
# 2. DATA PREPARATION FOR SAGEMAKER
# =================================================================

def prepare_datasets():
    """Transforms raw data into SageMaker-ready format"""
    # Get data paths
    data_dir = download_imdb_data()
    
    # Read and organize data
    data, labels = read_imdb_data(data_dir)
    
    # Combine positive/negative reviews and shuffle
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    
    # Preprocess text → convert reviews to cleaned word lists
    train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
    
    # Convert words to numerical features (Bag-of-Words)
    # We select 5000 most frequent words as our features
    train_X, test_X, vocabulary = extract_BoW_features(train_X, test_X, vocabulary_size=5000)
    
    # Create local directory for processed data
    data_dir = 'data/xgboost'
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert to pandas DataFrames
    train_X_df = pd.DataFrame(train_X)
    test_X_df = pd.DataFrame(test_X)
    train_y_series = pd.Series(train_y).astype(int)
    test_y_series = pd.Series(test_y).astype(int)

    # Split training data into training/validation sets
    # 10,000 training examples, 2,500 validation examples
    train_X = train_X_df.iloc[:10000].reset_index(drop=True)
    val_X = train_X_df.iloc[10000:12500].reset_index(drop=True)
    train_y = train_y_series.iloc[:10000].reset_index(drop=True)
    val_y = train_y_series.iloc[10000:12500].reset_index(drop=True)

    # Save datasets as CSV files
    # SageMaker requires specific formats:
    # - Training/validation: Label in first column, features in rest
    # - Test: Only features (no labels)
    test_X_df.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
    pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
    pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
    
    return data_dir

# =================================================================
# 3. SAGEMAKER SETUP & DATA UPLOAD
# =================================================================

def upload_to_s3(data_dir):
    """Uploads processed data to AWS S3 cloud storage"""
    # Initialize SageMaker session - our gateway to AWS ML services
    session = sagemaker.Session()
    
    # S3 bucket - cloud storage location (use your own bucket name)
    s3_bucket = "sagemaker-us-east-1-123456789012"
    
    # S3 prefix - folder name for organizing files
    prefix = 'imdb-sentiment'
    
    # Upload data to S3
    # SageMaker will automatically access this data during training
    data_location = session.upload_data(
        path=data_dir,    # Local folder with our CSV files
        bucket=s3_bucket, # Our cloud storage bucket
        key_prefix=prefix # Folder name in the bucket
    )
    
    print(f"📤 Data uploaded to: s3://{s3_bucket}/{prefix}/")
    return data_location

# =================================================================
# 4. MODEL TRAINING ON SAGEMAKER
# =================================================================

def train_model(data_location):
    """Launches XGBoost training job on SageMaker"""
    # Get SageMaker execution role - permissions for accessing AWS resources
    role = get_execution_role()
    
    # Retrieve XGBoost container image - preconfigured ML environment
    container = sagemaker.image_uris.retrieve('xgboost', session.boto_region_name, 'latest')
    
    # Configure training job:
    xgb = sagemaker.estimator.Estimator(
        image_uri=container,  # Pre-built XGBoost environment
        role=role,            # Permissions to access S3/other AWS services
        instance_count=1,     # Use one ML-optimized server
        instance_type='ml.m5.xlarge',  # General purpose instance ($0.23/hr)
        output_path=f's3://{s3_bucket}/{prefix}/output',  # Save model here
        sagemaker_session=session,
        
        # Cost optimization settings:
        use_spot_instances=True,  # Use discounted spot instances (save 70%)
        max_run=3600,             # Max training time (1 hour)
        max_wait=7200,            # Max wait for spot instance (2 hours)
        
        # Model hyperparameters:
        hyperparameters={
            'max_depth': 5,           # Control model complexity
            'eta': 0.2,               # Learning rate
            'objective': 'binary:logistic',  # Binary classification
            'num_round': 100          # Training iterations
        }
    )
    
    # Specify data locations in S3
    s3_input_train = sagemaker.TrainingInput(
        s3_data=f"{data_location}/train.csv", 
        content_type='csv'
    )
    s3_input_val = sagemaker.TrainingInput(
        s3_data=f"{data_location}/validation.csv", 
        content_type='csv'
    )
    
    # Launch training job
    xgb.fit({'train': s3_input_train, 'validation': s3_input_val})
    
    return xgb

# =================================================================
# 5. MODEL DEPLOYMENT & INFERENCE
# =================================================================

def deploy_endpoint(trained_model):
    """Deploys trained model to a real-time API endpoint"""
    # Create transformer for batch predictions
    batch_transformer = trained_model.transformer(
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{s3_bucket}/{prefix}/predictions'
    )
    
    # Run batch prediction on test set
    batch_transformer.transform(
        f"{data_location}/test.csv", 
        content_type='text/csv', 
        split_type='Line'
    )
    batch_transformer.wait()
    print(f"✅ Batch predictions saved to S3")
    
    # Deploy real-time endpoint
    predictor = trained_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',  # Low-cost endpoint ($0.05/hr)
        endpoint_name='imdb-sentiment-endpoint'
    )
    print(f"🚀 Real-time endpoint deployed!")
    return predictor

# =================================================================
# 6. MAIN EXECUTION FLOW
# =================================================================

if __name__ == "__main__":
    # 1. Prepare datasets locally
    data_dir = prepare_datasets()
    
    # 2. Upload data to cloud storage (S3)
    data_location = upload_to_s3(data_dir)
    
    # 3. Train model on SageMaker
    model = train_model(data_location)
    
    # 4. Deploy for predictions
    predictor = deploy_endpoint(model)
    
    # 5. Example prediction
    sample_review = "This movie was absolutely fantastic!"
    prediction = predictor.predict(sample_review)
    print(f"Predicted sentiment: {prediction}")
