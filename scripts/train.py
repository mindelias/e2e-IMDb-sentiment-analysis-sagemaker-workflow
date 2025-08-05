import os
import argparse
import pandas as pd
import xgboost as xgb
import subprocess
import sys


def install_packages():
    """Install required packages at runtime"""
    packages = ['sagemaker-training']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    
   # Install required packages first
    print("🔧 Installing required packages...")
    install_packages()
    
    # Now import sagemaker_training AFTER installation
    from sagemaker_training import environment
    
    env = environment.Environment()
    
    # Load data
    train_df = pd.read_csv(os.path.join(env.channel_input_dirs['train'], 'train.csv'), header=None)
    val_df = pd.read_csv(os.path.join(env.channel_input_dirs['validation'], 'validation.csv'), header=None)
    
    # Prepare data
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_val = val_df.iloc[:, 0]
    X_val = val_df.iloc[:, 1:]
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    params = {
        'max_depth': int(env.hyperparameters.get('max_depth', 5)),
        'eta': float(env.hyperparameters.get('eta', 0.2)),
        'gamma': float(env.hyperparameters.get('gamma', 4)),
        'min_child_weight': int(env.hyperparameters.get('min_child_weight', 6)),
        'subsample': float(env.hyperparameters.get('subsample', 0.8)),
        'objective': 'binary:logistic',
        'eval_metric': 'error'
    }
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dval, 'validation')],
        early_stopping_rounds=int(env.hyperparameters.get('early_stopping_rounds', 10)),
        num_boost_round=int(env.hyperparameters.get('num_round', 100))
    )
    
    # Save model
    model.save_model(os.path.join(env.model_dir, 'xgboost-model'))

if __name__ == "__main__":
    main()