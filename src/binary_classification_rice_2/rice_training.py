# -*- coding: utf-8 -*-
"""Main entry point for training binary classification models for the rice dataset."""

import argparse
import os
import sys
import keras

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the refactored classes and utilities
from src.common.types import ExperimentSettings
from src.common import csv_utils
from src.binary_classification_rice_2.rice_data_wrapper import RiceDataWrapper
from src.binary_classification_rice_2.logistic_regression_trainer import LogisticRegressionTrainer

# Set the random seeds for reproducibility
keras.utils.set_random_seed(42)

# Configuration for the model, using our common ExperimentSettings type
MODEL_CONFIGS = {
    "rice_model": ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        input_features=['Eccentricity', 'Major_Axis_Length', 'Area'],
        verbose=1,
    ),
}

def main():
    """Orchestrates the model training process."""
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train a specified classification model.")
    parser.add_argument(
        "model_type",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        default="rice_model",
        nargs='?',
        help=f"The type of model to train. Choose from: {list(MODEL_CONFIGS.keys())}",
    )
    args = parser.parse_args()
    model_type = args.model_type

    # 2. Setup Paths and Configuration
    print(f"\n--- Starting process for model: {model_type} ---")
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    settings = MODEL_CONFIGS[model_type]

    # Define paths for processed data
    train_csv_path = os.path.join(artifacts_dir, "rice_train.csv")
    val_csv_path = os.path.join(artifacts_dir, "rice_validation.csv")
    test_csv_path = os.path.join(artifacts_dir, "rice_test.csv")

    # 3. Conditional Data Loading/Processing
    if all(os.path.exists(p) for p in [train_csv_path, val_csv_path, test_csv_path]):
        print("\nFound pre-processed data. Loading directly from CSV files...")
        train_df = csv_utils.load_csv(train_csv_path)
        val_df = csv_utils.load_csv(val_csv_path)
        test_df = csv_utils.load_csv(test_csv_path)
    else:
        print("\nProcessed data not found. Starting data processing pipeline...")
        input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')
        data_wrapper = RiceDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
        try:
            train_df, val_df, test_df = data_wrapper.process_and_split()
        except (FileNotFoundError, ValueError) as e:
            print(f"Error during data processing: {e}")
            print("Aborting.")
            return

    if train_df is None or val_df is None or test_df is None:
        print("Error: Data loading failed. Aborting.")
        return

    # 4. Model Training using the refactored Trainer
    trainer = LogisticRegressionTrainer(experiment_name=model_type, settings=settings)
    trainer.build_model()
    trainer.train(train_df, val_df)
    trainer.evaluate(test_df)
    trainer.save(artifacts_dir)

    print(f"\n--- Successfully completed all steps for model: {model_type} ---")

if __name__ == "__main__":
    main()
