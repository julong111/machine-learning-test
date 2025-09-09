# -*- coding: utf-8 -*-
"""Main entry point for training linear regression models for the taxi dataset."""

import argparse
import os

# Import the new refactored classes and types
from src.common.types import ExperimentSettings
from src.common import csv_utils
from src.linear_regression_taxi_1.taxi_data_wrapper import TaxiDataWrapper
from src.linear_regression_taxi_1.linear_regression_trainer import LinearRegressionTrainer

# Configuration for all models, using our common ExperimentSettings type
MODEL_CONFIGS = {
    "single_feature": ExperimentSettings(
        learning_rate=0.01,
        number_epochs=20,
        batch_size=1000,
        input_features=["TRIP_MILES"],
        verbose=1,
    ),
    "multi_feature": ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
        input_features=["TRIP_MILES", "TRIP_MINUTES"],
        verbose=1,
    ),
    "smart_model": ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
        input_features=["TRIP_MILES", "ESTIMATED_MINUTES"],
        verbose=1,
    ),
}

def main():
    """Orchestrates the model training process."""
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train a specified regression model.")
    parser.add_argument(
        "model_type",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        help=f"The type of model to train. Choose from: {list(MODEL_CONFIGS.keys())}",
    )
    args = parser.parse_args()
    model_type = args.model_type

    # 2. Setup Paths and Configuration
    print(f"\n--- Starting process for model: {model_type} ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')
    settings = MODEL_CONFIGS[model_type]

    # Define paths for processed data
    train_csv_path = os.path.join(artifacts_dir, "chicago_taxi_train.csv")
    val_csv_path = os.path.join(artifacts_dir, "chicago_taxi_validation.csv")
    test_csv_path = os.path.join(artifacts_dir, "chicago_taxi_test.csv")

    # 3. Conditional Data Loading/Processing
    if all(os.path.exists(p) for p in [train_csv_path, val_csv_path, test_csv_path]):
        print("\nFound pre-processed data. Loading directly from CSV files...")
        train_df = csv_utils.load_csv(train_csv_path)
        val_df = csv_utils.load_csv(val_csv_path)
        test_df = csv_utils.load_csv(test_csv_path)
    else:
        print("\nProcessed data not found. Starting data processing pipeline...")
        # Path to the raw data is needed for processing
        input_csv_path = os.path.join(artifacts_dir, 'chicago_taxi_google.csv')
        data_wrapper = TaxiDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
        try:
            train_df, val_df, test_df = data_wrapper.process_and_split()
        except FileNotFoundError as e:
            print(f"Error during data processing: {e}")
            print("Aborting.")
            return

    # Check if dataframes are valid before proceeding
    if train_df is None or val_df is None or test_df is None:
        print("Error: Data loading failed. One or more dataframes are None.")
        print(f"Please ensure data files exist and are valid in {artifacts_dir}")
        return

    # 4. Model Training
    trainer = LinearRegressionTrainer(experiment_name=model_type, settings=settings)
    trainer.build_model()
    trainer.train(train_df, val_df)
    trainer.evaluate(test_df)
    trainer.save(artifacts_dir)

    print(f"\n--- Successfully completed all steps for model: {model_type} ---")

if __name__ == "__main__":
    main()
