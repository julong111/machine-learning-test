# -*- coding: utf-8 -*-
"""
Unified training script for all regression models.

This script serves as the single entry point for training different models.
Use command-line arguments to specify which model to train.

Example Usage:
    python src/linear_regression_taxi_1/train.py single_feature
    python src/linear_regression_taxi_1/train.py multi_feature
    python src/linear_regression_taxi_1/train.py smart_model
"""

import argparse
import json
from pathlib import Path

from src.linear_regression_taxi_1 import utils, pipeline

# --- Configuration for all models ---
# A dictionary to hold the settings for each model type.
MODEL_CONFIGS = {
    "single_feature": utils.ExperimentSettings(
        learning_rate=0.01,
        number_epochs=20,
        batch_size=1000,
        input_features=["TRIP_MILES"],
        verbose=1,
    ),
    "multi_feature": utils.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
        input_features=["TRIP_MILES", "TRIP_MINUTES"],
        verbose=1,
    ),
    "smart_model": utils.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
        input_features=["TRIP_MILES", "ESTIMATED_MINUTES"],
        verbose=1,
    ),
}


# --- Special logic for 'smart_model' ---
# This class is only used by the 'smart_model', so we keep it co-located here.
class DurationEstimator:
    """Loads a lookup table to provide trip duration estimates based on distance."""
    def __init__(self, config_path: Path):
        print(f"Loading duration estimation rules from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                self.lookup_table = {float(k): v for k, v in json.load(f).items()}
            self.sorted_upper_bounds = sorted(self.lookup_table.keys())
            print("Duration estimator initialized successfully.")
        except FileNotFoundError:
            print(f"Error: Lookup table not found at {config_path}")
            print("Please run 'analyze_time_distance.py' first to generate the lookup table.")
            raise

    def estimate(self, miles: float) -> float:
        """Estimates trip duration by finding the correct bin in the loaded lookup table."""
        for upper_bound in self.sorted_upper_bounds:
            if miles < upper_bound:
                return self.lookup_table[upper_bound]
        return self.lookup_table[self.sorted_upper_bounds[-1]]


def run_feature_engineering_for_smart_model(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Applies the specific feature engineering steps for the smart_model."""
    print("\nEngineering 'ESTIMATED_MINUTES' feature for smart_model...")
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    lookup_table_path = artifacts_dir / "time_distance_lookup.json"

    try:
        estimator = DurationEstimator(lookup_table_path)
    except FileNotFoundError:
        # We should stop execution if the required file is missing.
        raise

    df["ESTIMATED_MINUTES"] = df["TRIP_MILES"].apply(estimator.estimate)
    print("Feature engineering complete.\n")
    return df


# --- Main Execution Logic ---
def main():
    """Main function to parse arguments and run the specified training."""
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

    # 2. Get Configuration
    print(f"\n--- Starting training for model: {model_type} ---")
    settings = MODEL_CONFIGS[model_type]
    experiment_name = model_type # The experiment name is the same as the model type

    # 3. Load Data
    training_df = pipeline.load_and_prepare_data(
        "artifacts/linear_regression_taxi_1/chicago_taxi_train.csv"
    )

    # 4. Conditional Feature Engineering
    if model_type == "smart_model":
        try:
            training_df = run_feature_engineering_for_smart_model(training_df)
        except FileNotFoundError:
            # Stop the process if feature engineering fails
            print("Aborting training due to missing feature engineering resources.")
            return

    # 5. Run Experiment
    pipeline.run_experiment(
        experiment_name=experiment_name,
        settings=settings,
        training_df=training_df,
        label_name="FARE",
    )
    print(f"--- Successfully completed training for model: {model_type} ---")


if __name__ == "__main__":
    main()
