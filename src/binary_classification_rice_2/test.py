# -*- coding: utf-8 -*-
"""Main entry point for evaluating binary classification models for the rice dataset."""

import argparse
import os
import sys
import keras
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the refactored classes and utilities
from src.common import csv_utils
from src.common.keras_utils import prepare_features_and_labels

# Configuration for the model features, must match training
# This should ideally be in a shared config file
MODEL_CONFIGS = {
    "rice_model": {
        "input_features": ['Eccentricity', 'Major_Axis_Length', 'Area'],
    }
}

def main():
    """Orchestrates the model evaluation process."""
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model.")
    parser.add_argument(
        "model_type",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        default="rice_model",
        nargs='?',
        help=f"The type of model to evaluate. Choose from: {list(MODEL_CONFIGS.keys())}",
    )
    args = parser.parse_args()
    model_type = args.model_type

    # 2. Setup Paths and Configuration
    print(f"\n--- Starting evaluation for model: {model_type} ---")
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    
    # Define paths for test data and model
    test_csv_path = os.path.join(artifacts_dir, "rice_test.csv")
    model_path = os.path.join(artifacts_dir, f'{model_type}.keras')

    if not os.path.exists(test_csv_path):
        print(f"Error: Test data not found at {test_csv_path}")
        print("Please run the training script first to generate the data.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run the training script first to generate the model.")
        return

    # 3. Load Test Data
    print(f"Loading test data from {test_csv_path}...")
    test_df = csv_utils.load_csv(test_csv_path)
    if test_df is None:
        print("Error: Failed to load test data. Aborting.")
        return

    # 4. Feature and Label Preparation
    config = MODEL_CONFIGS[model_type]
    input_features = config['input_features']
    label_column = 'Class_Bool'
    
    test_features, test_labels = prepare_features_and_labels(
        test_df, input_features, label_column
    )

    # 5. Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 6. Model Evaluation
    print("\n--- Evaluating Model Performance on Test Set ---")
    metrics = model.evaluate(test_features, test_labels, verbose=1)
    
    print("\n--- Evaluation Complete ---")
    print("Test Metrics:")
    for metric_name, metric_value in zip(model.metrics_names, metrics):
        print(f"  {metric_name}: {metric_value:.4f}")

    print(f"\n--- Successfully completed evaluation for model: {model_type} ---")


if __name__ == "__main__":
    main()
