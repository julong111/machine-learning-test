# -*- coding: utf-8 -*-
"""Main entry point for training linear regression models for the taxi dataset."""

import argparse
import os
import json

# Import the new refactored classes and types
from src.common.types import ExperimentSettings
from src.common import csv_utils
from src.linear_regression_taxi_1.linear_regression_trainer import LinearRegressionTrainer

# Configuration for all models, using our common ExperimentSettings type
MODEL_CONFIGS = {
    "single_feature": ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
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
        # These are now fallback values if the tuning file is not found.
        learning_rate=0.072250,
        number_epochs=20,
        batch_size=32,
        input_features=["TRIP_MILES", "ESTIMATED_MINUTES", "TRIP_MILES_SQ"],
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

    # --- Load Tuned Hyperparameters (if available) ---
    # This makes the training script "tuning-aware".
    if model_type == "smart_model":
        hyperparams_path = os.path.join(artifacts_dir, 'best_hyperparameters.json')
        if os.path.exists(hyperparams_path):
            print(f"\nFound best hyperparameters file at {hyperparams_path}. Overriding defaults.")
            with open(hyperparams_path, 'r') as f:
                best_params = json.load(f)
            # Update the settings object with the tuned values
            settings.learning_rate = best_params.get('learning_rate', settings.learning_rate)
            settings.batch_size = best_params.get('batch_size', settings.batch_size)
            print(f"Using tuned learning_rate: {settings.learning_rate}, batch_size: {settings.batch_size}")
        else:
            print("\nNo tuned hyperparameters file found. Using default values from configuration.")

    # Define paths for processed data
    train_csv_path = os.path.join(artifacts_dir, "chicago_taxi_train.csv")
    val_csv_path = os.path.join(artifacts_dir, "chicago_taxi_validation.csv")

    # 3. Conditional Data Loading/Processing
    if all(os.path.exists(p) for p in [train_csv_path, val_csv_path]):
        print("\nFound pre-processed data. Loading directly from CSV files...")
        train_df = csv_utils.load_csv(train_csv_path)
        val_df = csv_utils.load_csv(val_csv_path)
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
    if train_df is None or val_df is None:
        print("Error: Data loading failed. One or more dataframes are None.")
        print(f"Please ensure data files exist and are valid in {artifacts_dir}")
        return

    # 4. Model Training
    trainer = LinearRegressionTrainer(experiment_name=model_type, settings=settings)
    trainer.build_model()
    trainer.train(train_df, val_df)
    trainer.save(artifacts_dir)

    print(f"\n--- Successfully completed all steps for model: {model_type} ---")

if __name__ == "__main__":
    main()
