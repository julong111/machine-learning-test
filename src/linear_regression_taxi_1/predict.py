# -*- coding: utf-8 -*-
"""
Unified prediction script for all regression models.

This script serves as the single entry point for making predictions.
Use command-line arguments to specify which model to use and to provide
the necessary features.

Example Usage:
    # Single-feature model
    python src/linear_regression_taxi_1/predict.py single_feature --trip-miles 10.5

    # Multi-feature model
    python src/linear_regression_taxi_1/predict.py multi_feature --trip-miles 10.5 --trip-minutes 25

    # Smart model
    python src/linear_regression_taxi_1/predict.py smart_model --trip-miles 10.5
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import keras

# --- Special logic for 'smart_model' ---
# This class is copied from train.py as it's needed for prediction as well.
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


def main():
    """Main function to parse arguments and run the specified prediction."""
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Make a prediction using a trained regression model.")
    subparsers = parser.add_subparsers(dest="model_type", required=True, help="The type of model to use for prediction.")

    # Sub-parser for the single_feature model
    parser_single = subparsers.add_parser("single_feature", help="Predict using the single-feature model.")
    parser_single.add_argument("--trip-miles", type=float, required=True, help="The distance of the trip in miles.")

    # Sub-parser for the multi_feature model
    parser_multi = subparsers.add_parser("multi_feature", help="Predict using the multi-feature model.")
    parser_multi.add_argument("--trip-miles", type=float, required=True, help="The distance of the trip in miles.")
    parser_multi.add_argument("--trip-minutes", type=float, required=True, help="The duration of the trip in minutes.")

    # Sub-parser for the smart_model
    parser_smart = subparsers.add_parser("smart_model", help="Predict using the smart model with feature engineering.")
    parser_smart.add_argument("--trip-miles", type=float, required=True, help="The distance of the trip in miles.")

    args = parser.parse_args()
    model_type = args.model_type

    # --- 2. Load Model and Settings ---
    print(f"\n--- Making prediction with model: {model_type} ---")
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    model_path = artifacts_dir / f"{model_type}_model.keras"
    settings_path = artifacts_dir / f"{model_type}_settings.pkl"

    try:
        model = keras.models.load_model(model_path)
        with open(settings_path, "rb") as f:
            settings = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or settings for '{model_type}': {e}")
        print(f"Please ensure you have run the training script first: 'python src/linear_regression_taxi_1/train.py {model_type}'")
        return

    # --- 3. Prepare Input Data (with conditional feature engineering) ---
    input_data = {}
    print_data = {} # For displaying a clean summary to the user

    if model_type == "smart_model":
        print_data["Trip Miles"] = args.trip_miles
        try:
            lookup_table_path = artifacts_dir / "time_distance_lookup.json"
            estimator = DurationEstimator(lookup_table_path)
            estimated_minutes = estimator.estimate(args.trip_miles)
            print(f"Estimated trip duration: {estimated_minutes:.2f} minutes")
            print_data["Estimated Minutes"] = estimated_minutes
            
            input_data["TRIP_MILES"] = np.array([args.trip_miles])
            input_data["ESTIMATED_MINUTES"] = np.array([estimated_minutes])

        except FileNotFoundError:
            print("Aborting prediction due to missing feature engineering resources.")
            return
            
    elif model_type == "multi_feature":
        print_data["Trip Miles"] = args.trip_miles
        print_data["Trip Minutes"] = args.trip_minutes
        input_data["TRIP_MILES"] = np.array([args.trip_miles])
        input_data["TRIP_MINUTES"] = np.array([args.trip_minutes])

    elif model_type == "single_feature":
        print_data["Trip Miles"] = args.trip_miles
        input_data["TRIP_MILES"] = np.array([args.trip_miles])

    # --- 4. Make Prediction ---
    prediction = model.predict(input_data, verbose=0)
    predicted_fare = prediction[0][0]

    # --- 5. Display Results ---
    print("\n--- Prediction Summary ---")
    for key, value in print_data.items():
        print(f"{key}: {value:.2f}")
    print("--------------------------")
    print(f"Predicted Fare: ${predicted_fare:.2f}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
