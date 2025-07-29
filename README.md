# IMDb Sentiment Analysis with SageMaker - End-to-End Workflow
https://d1.awsstatic.com/Projects/P5505030/architecture-diagram_Project_500x302.7d1fe3d5d8b5b61b8e3f4f8e3b8e3f4f8e3f4f8e3.png

📖 Overview
This project demonstrates a production-ready machine learning pipeline for sentiment analysis on IMDb movie reviews using AWS SageMaker. The workflow includes data preprocessing, distributed training with XGBoost, real-time deployment, and batch predictions - showcasing industry best practices for MLOps.

✨ Key Features
Automated Data Processing: Clean HTML, remove stopwords, and convert text to numerical features

Cost-Optimized Training: 70% savings using Spot Instances with checkpointing

Two Deployment Options: Real-time API endpoints + Batch Transform

Production Monitoring: CloudWatch integration for performance tracking

Reproducible Workflows: SageMaker Pipelines for CI/CD readiness



# 📂 Repository Structure
imdb-sagemaker-sentiment/
├── data/                   # Raw and processed datasets
│   ├── aclImdb/            # Original IMDb dataset
│   └── processed/          # Cleaned CSV files
├── notebooks/              # Jupyter notebooks for exploration
│   └── sentiment_analysis.ipynb
├── scripts/                # Processing and training scripts
│   ├── preprocessing.py    # Text cleaning functions
│   ├── processing_job.py   # SageMaker processing script
│   └── train.py            # XGBoost training script
├── utils/                  # Utility functions
│   └── test.py             # Endpoint testing utility
├── pipeline.py             # Main orchestration script
├── requirements.txt        # Python dependencies
└── README.md               # This documentation


# 🛠️ Setup Instructions (Choose Your Path)
## Option 1: AWS SageMaker Notebook Instance (Recommended)

    1. Create Notebook Instance:

        - Go to SageMaker Console
        - Create new notebook instance:
            - Instance type: ml.t3.medium
            - IAM role: AmazonSageMakerFullAccess
            - Volume size: 20GB

    2. Clone Repository:
        - git clone https://github.com/username/imdb-sagemaker-sentiment.git
        - cd imdb-sagemaker-sentiment

## Option 2: Local Machine Setup
    1. Prerequisites:
        - Python 3.8+
        - AWS CLI configured (aws configure)
        - Docker (for local processing 

    2. Create Virtual Environment:
        - python -m venv .venv
        - source .venv/bin/activate  # Linux/Mac
        - .venv\Scripts\activate    # Windows

    3. Install Dependencies:
        - pip install -r requirements-local.txt
        - python -c "import nltk; nltk.download('stopwords')"

    4. Run Pipeline:
        **For full cloud execution**
        - python pipeline.py --mode cloud

        **For local testing (small dataset)**
        - python pipeline.py --mode local --sample-size 1000


# 🧪 TESR YOUR SETUP
## Cloud/Hybrid Mode:
**In notebook or local script**
`from utils.test import test_endpoint`
# Test with positive review
print(test_endpoint(
    endpoint_name='imdb-sentiment-api',
    review_text="This movie was breathtaking! The cinematography and acting were phenomenal."
))

# Test with negative review
print(test_endpoint(
    endpoint_name='imdb-sentiment-api',
    review_text="I've never been more disappointed. The plot was confusing and the acting wooden."
))

## Local Mode:
python local_api.py  # Starts Flask server at http://localhost:5000
curl -X POST http://localhost:5000/predict -d "review=This film disappointed me"
# Output: {"prediction":0.12,"sentiment":"NEGATIVE"}

