import boto3
import joblib
import pandas as pd
import numpy as np
from io import BytesIO



def test_endpoint(endpoint_name, review_text, vectorizer_path):
    from scripts.preprocessing import review_to_words  # Make sure this is accessible
    """
    Tests a SageMaker endpoint with a sample review
    
    Args:
        endpoint_name (str): Name of the deployed endpoint
        review_text (str): Raw review text to test
        vectorizer_path (str): S3 path to the vectorizer file
        
    Returns:
        dict: Prediction result with probability and sentiment
    """
    # 1. Preprocess text
    print("🧹 Preprocessing text...")
    cleaned_text = review_to_words(review_text)
    
    # 2. Load vectorizer from S3
    print("📥 Loading vectorizer...")
    s3 = boto3.client('s3')
    bucket, key = vectorizer_path.replace('s3://', '').split('/', 1)
    
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        vectorizer = joblib.load(BytesIO(response['Body'].read()))
        print(f"✅ Loaded vectorizer from s3://{bucket}/{key}")
    except Exception as e:
        return {
            'error': f"Vectorizer loading failed: {str(e)}",
            'message': f'Check path: s3://{bucket}/{key}'
        }
    
    # 3. Transform text to features
    print("🔢 Transforming text to features...")
    try:
        features = vectorizer.transform([cleaned_text]).toarray()[0]
        features_str = ",".join(map(str, features))
    except Exception as e:
        return {
            'error': f"Text transformation failed: {str(e)}",
            'message': 'Check vectorizer compatibility'
        }
    
    # 4. Create SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')
    
    try:
        # 5. Send request to endpoint
        print("📤 Sending request to endpoint...")
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=features_str
        )
        
        # 6. Parse prediction
        probability = float(response['Body'].read())
        sentiment = "POSITIVE" if probability > 0.5 else "NEGATIVE"
        
        return {
            'prediction': probability,
            'sentiment': sentiment,
            'review': review_text
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Make sure endpoint exists and is InService'
        }