#!/bin/bash

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r /opt/ml/processing/code/requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords', download_dir='/tmp/nltk_data')"

# Run the actual processing script
echo "Starting processing job..."
python /opt/ml/processing/input/code/processing_job.py "$@"
