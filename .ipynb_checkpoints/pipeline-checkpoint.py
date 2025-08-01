import boto3
import os
import glob
from pathlib import Path
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost import XGBoost, XGBoostPredictor

def download_imdb_data():
    """Download IMDb dataset if not already present"""
    data_dir = Path('../data/aclImdb')
    if not data_dir.exists():
        print("Downloading IMDb dataset...")
        # In production, we'd use a more reliable source
        os.system('wget -P ../data/ https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
        os.system('tar -xvzf ../data/aclImdb_v1.tar.gz -C ../data/')
    return str(data_dir)

def create_processing_step(role):
    """Create SageMaker Processing Job"""
    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type='ml.m5.large',
        instance_count=1,
        base_job_name='imdb-processing'
    )
    
    return ProcessingStep(
        name='IMDBDataProcessing',
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=download_imdb_data(),
                destination='/opt/ml/processing/input/data'
            )
        ],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/output/train'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/output/validation'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/output/test'),
            ProcessingOutput(output_name='vectorizer', source='/opt/ml/processing/output/vectorizer')
        ],
        code='scripts/processing_job.py'
    )

def create_training_step(train_s3, val_s3, role):
    """Create SageMaker Training Job"""
    container = sagemaker.image_uris.retrieve('xgboost', region, '1.7-1')
    
    hyperparameters = {
        'max_depth': '5',
        'eta': '0.2',
        'gamma': '4',
        'min_child_weight': '6',
        'subsample': '0.8',
        'objective': 'binary:logistic',
        'early_stopping_rounds': '10',
        'num_round': '100'
    }
    
    xgb_estimator = XGBoost(
        entry_point='train.py',
        hyperparameters=hyperparameters,
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type='ml.m5.2xlarge',
        framework_version='1.7-1',
        output_path=f's3://{bucket}/{prefix}/models',
        use_spot_instances=True,
        max_wait=7200,
        max_run=3600
    )
    
    return TrainingStep(
        name='IMDBModelTraining',
        estimator=xgb_estimator,
        inputs={
            'train': TrainingInput(s3_data=train_s3, content_type='text/csv'),
            'validation': TrainingInput(s3_data=val_s3, content_type='text/csv')
        }
    )

def main():
    # Initialize SageMaker
    session = sagemaker.Session()
    role = get_execution_role()
    region = session.boto_region_name
    bucket =  "sentiment-analysis-workflow"
    prefix = 'imdb-sentiment-analysis'
    
    processing_step = create_processing_step(role)
    
    train_s3 = processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri
    val_s3 = processing_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri
    training_step = create_training_step(train_s3, val_s3, role)
    
    pipeline = Pipeline(
        name='IMDBSentimentPipeline',
        steps=[processing_step, training_step],
        sagemaker_session=session
    )
    
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()
    
    model_data = execution.steps[1].properties.ModelArtifacts.S3ModelArtifacts
    endpoint_name = 'imdb-sentiment-endpoint'
    
    # Deploy model
    model = sagemaker.model.Model(
        image_uri=sagemaker.image_uris.retrieve('xgboost', region, '1.7-1'),
        model_data=model_data,
        role=role,
        predictor_cls=XGBoostPredictor
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name
    )
    
    print("✅ End-to-end workflow completed!")

if __name__ == "__main__":
    main()