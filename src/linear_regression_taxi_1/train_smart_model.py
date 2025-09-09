# -*- coding: utf-8 -*-
"""Trains a 'smart' model by loading an external configuration for feature engineering."""

import json
from pathlib import Path
from src.linear_regression_taxi_1 import utils, pipeline


# This class will load the lookup table and provide the estimation logic.
class DurationEstimator:
    def __init__(self, config_path: Path):
        print(f"Loading duration estimation rules from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                # The keys are strings, so we convert them to float for comparison
                self.lookup_table = {float(k): v for k, v in json.load(f).items()}
            # Sort the keys to make searching easier and more reliable
            self.sorted_upper_bounds = sorted(self.lookup_table.keys())
            print("Duration estimator initialized successfully.")
        except FileNotFoundError:
            print(f"Error: Lookup table not found at {config_path}")
            print("Please run 'analyze_time_distance.py' first to generate the lookup table.")
            raise

    def estimate(self, miles: float) -> float:
        """Estimates trip duration by finding the correct bin in the loaded lookup table."""
        # Find the first upper bound that is greater than the given miles
        for upper_bound in self.sorted_upper_bounds:
            if miles < upper_bound:
                return self.lookup_table[upper_bound]
        # If miles is greater than all upper bounds, it belongs in the last bin
        # The last key in a sorted dictionary is the largest.
        return self.lookup_table[self.sorted_upper_bounds[-1]]

def main():
    """Defines and runs the smart model experiment using the external lookup table."""
    # --- Configuration ---
    # Paths are relative to the project root, where the script should be run from.
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    lookup_table_path = artifacts_dir / "time_distance_lookup.json"

    experiment_name = "smart_model"
    settings = utils.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=20,
        batch_size=50,
        input_features=["TRIP_MILES", "ESTIMATED_MINUTES"],
        verbose=1,
    )

    # --- Initialization ---
    # Initialize the estimator, which loads the JSON file.
    try:
        estimator = DurationEstimator(lookup_table_path)
    except FileNotFoundError:
        return # Stop execution if the lookup table is not found

    # --- Data Loading and Feature Engineering ---
    # Load data from the correct new path
    training_df = pipeline.load_and_prepare_data(
        "artifacts/linear_regression_taxi_1/chicago_taxi_train.csv"
    )
    print("\nEngineering 'ESTIMATED_MINUTES' feature using the loaded lookup table...")
    training_df["ESTIMATED_MINUTES"] = training_df["TRIP_MILES"].apply(estimator.estimate)
    print("Feature engineering complete.\n")

    # --- Model Training ---
    pipeline.run_experiment(
        experiment_name=experiment_name,
        settings=settings,
        training_df=training_df,
        label_name="FARE",
    )


if __name__ == "__main__":
    main()
