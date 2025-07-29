import boto3
import joblib
from io import BytesIO
from scripts.preprocessing import review_to_words

def test_endpoint(endpoint_name, review_text, vectorizer_path):
    """
    Tests a SageMaker endpoint with a sample review
    
    Steps:
    1. Preprocess the review text
    2. Load vectorizer from S3
    3. Transform text to BoW features
    4. Send to endpoint
    
    Returns:
        dict: Prediction result with probability and sentiment
    """
    # 1. Preprocess text
    words = review_to_words(review_text)
    
    # 2. Load vectorizer from S3
    s3 = boto3.client('s3')
    bucket, key = vectorizer_path.replace('s3://', '').split('/', 1)
    response = s3.get_object(Bucket=bucket, Key=key)
    vectorizer = joblib.load(BytesIO(response['Body'].read()))
    
    # 3. Transform to BoW features
    bow_features = vectorizer.transform([words]).toarray()[0]
    features_str = ",".join(map(str, bow_features))
    
    # 4. Create SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')
    
    try:
        # 5. Send request to endpoint
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