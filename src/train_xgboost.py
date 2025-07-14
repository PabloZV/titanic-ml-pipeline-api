"""
Script to train an XGBoost model on the Titanic dataset and save the model and feature importances.
Enhanced for best practices: config, logging, evaluation report, random seed, error handling, test set evaluation, modularization.
"""
import os
import sys
import pickle
import json
import logging
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.loaders import load_required_attributes_from_raw
from src.column_transformers import TitanicInputTransformer
from src.train_profiling import profile_stage
from src.logging_utils import setup_logging

def set_global_seed(seed=42):
    import random
    import numpy as np
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- Default Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, '../data/train.csv')
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/best_model.pkl')
DEFAULT_IMPORTANCES_PATH = os.path.join(SCRIPT_DIR, '../models/feature_importance.json')
DEFAULT_EVAL_REPORT_PATH = os.path.join(SCRIPT_DIR, '../models/eval_report.json')
DEFAULT_LOG_PATH = os.path.join(SCRIPT_DIR, '../logs/train_pipeline.log')

def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model for Titanic dataset.")
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON file (optional)')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH, help=f'Path to training data CSV (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help=f'Path to save trained model (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--importances-path', type=str, default=DEFAULT_IMPORTANCES_PATH, help=f'Path to save feature importances JSON (default: {DEFAULT_IMPORTANCES_PATH})')
    parser.add_argument('--eval-report-path', type=str, default=DEFAULT_EVAL_REPORT_PATH, help=f'Path to save evaluation report JSON (default: {DEFAULT_EVAL_REPORT_PATH})')
    parser.add_argument('--log-path', type=str, default=DEFAULT_LOG_PATH, help='Path to log file')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--logs-config-path', type=str, default=None, help='Path to logging config JSON (optional)')
    return parser.parse_args()


def save_importances_from_xgboost_model(model, X_train, importances_path):
    """
    Extract and save grouped feature importances from a fitted XGBoost pipeline.
    """
    input_transform = model.named_steps["input_transform"]
    scaler = model.named_steps.get("scaler")
    clf = model.named_steps["clf"]

    X_transformed = input_transform.fit_transform(X_train)
    if scaler is not None:
        scaler.fit(X_transformed)
        def get_feature_names_from_column_transformer(ct: ColumnTransformer):
            feature_names = []
            for name, transformer, cols in ct.transformers_:
                if transformer in ('drop', None):
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out(cols)
                else:
                    names = cols
                feature_names.extend(names)
            return feature_names
        feature_names = get_feature_names_from_column_transformer(scaler)
    else:
        feature_names = (
            X_transformed.columns.tolist()
            if isinstance(X_transformed, pd.DataFrame)
            else X_train.columns.tolist()
        )
    importances = clf.feature_importances_
    reverse_mapping = {
        'Age': 'Age',
        'Fare': 'Fare',
        'SibSp': 'SibSp',
        'Parch': 'Parch',
        'male': 'Sex',
        'female': 'Sex',
        'Class_1': 'Pclass',
        'Class_2': 'Pclass',
        'Class_3': 'Pclass',
        'C': 'Embarked',
        'Q': 'Embarked',
        'S': 'Embarked'
    }
    def map_feature_to_user_key(feature_name):
        return reverse_mapping.get(feature_name, feature_name)
    grouped_importances = defaultdict(list)
    for feat, imp in zip(feature_names, importances):
        user_key = map_feature_to_user_key(feat)
        grouped_importances[user_key].append(imp)
    df = pd.DataFrame([
        {"User Input Key": key, "Total Importance": round(float(np.sum(imps)), 4)}
        for key, imps in grouped_importances.items()
    ])
    df = df.sort_values("Total Importance", ascending=False).reset_index(drop=True)
    df.to_json(importances_path, orient="records", indent=2)

def save_evaluation_report(report, path):
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    args = parse_args()
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    # Paths
    data_path = config.get('data_path', args.data_path)
    model_path = config.get('model_path', args.model_path)
    importances_path = config.get('importances_path', args.importances_path)
    eval_report_path = config.get('eval_report_path', args.eval_report_path)
    log_path = config.get('log_path', args.log_path)
    test_size = config.get('test_size', args.test_size)
    random_seed = config.get('random_seed', args.random_seed)

    # Use logs_config_path from CLI/config, fallback to default location if not provided
    logs_config_path = args.logs_config_path
    if not logs_config_path:
        logs_config_path = os.path.join(SCRIPT_DIR, '../configs/logs_config.json')
    setup_logging(log_path, logs_config_path=logs_config_path)
    set_global_seed(random_seed)
    logging.info("Starting XGBoost training pipeline...")
    try:

        # Load data (profiled)
        X, y = profile_stage("Load Data", load_required_attributes_from_raw, data_path)
        logging.info(f"Loaded data from {data_path} with shape {X.shape}")

        # Split train/test (profiled)
        X_train, X_test, y_train, y_test = profile_stage(
            "Train/Test Split", train_test_split,
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        logging.info(f"Split data: train {X_train.shape}, test {X_test.shape}")

        # Preprocessing pipeline
        numeric_features = ['Age', 'Fare']
        numeric_scaler = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_features)],
            remainder='passthrough'
        )
        pipeline = Pipeline([
            ('input_transform', TitanicInputTransformer()),
            ('scaler', numeric_scaler),
            ('clf', xgb.XGBClassifier(eval_metric='logloss', random_state=random_seed))
        ])
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__subsample': [0.8, 1],
            'clf__colsample_bytree': [0.8, 1],
            'clf__gamma': [0, 1],
            'clf__min_child_weight': [1, 5]
        }
        scoring = {
            #'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
            
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            n_jobs=-1,
            return_train_score=True
        )

        logging.info("Training XGBoost model with GridSearchCV...")
        profile_stage("GridSearchCV Training", grid.fit, X_train, y_train)
        logging.info(f"Best parameters: {grid.best_params_}")
        logging.info(f"Best CV F1 score: {grid.best_score_}")

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(grid, f)
        logging.info(f"Model saved to {model_path}")

        # Save feature importances
        save_importances_from_xgboost_model(
            model=grid.best_estimator_,
            X_train=X_train,
            importances_path=importances_path
        )
        logging.info(f"Feature importances saved to {importances_path}")

        # Evaluate on test set (profiled)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        y_pred = profile_stage("Model Evaluation (Predict)", grid.predict, X_test)
        test_metrics = {
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        logging.info(f"Test set metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'classification_report'}, indent=2)}")

        # Save evaluation report (CV and test)
        eval_report = {
            'cv_best_params': grid.best_params_,
            'cv_best_score': grid.best_score_,
            'cv_results': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in grid.cv_results_.items()},
            'test_metrics': test_metrics
        }
        save_evaluation_report(eval_report, eval_report_path)
        logging.info(f"Evaluation report saved to {eval_report_path}")

    except Exception as e:
        logging.exception(f"Error during training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()