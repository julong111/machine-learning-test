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
import pandas as pd
import numpy as np
import keras

# Import the shared DurationEstimator from our new utility file
from src.linear_regression_taxi_1.taxi_duration_estimator import DurationEstimator


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

    # Load scaler if needed for the model
    scaler = None
    if model_type == "smart_model":
        scaler_path = artifacts_dir / 'fare_scaler.pkl'
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Feature scaler loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {scaler_path}. Please run the data preparation script first.")
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
            trip_miles_sq = args.trip_miles ** 2
            print_data["Estimated Minutes"] = estimated_minutes

            # Create a temporary DataFrame for scaling
            features_df = pd.DataFrame({
                "TRIP_MILES": [args.trip_miles],
                "ESTIMATED_MINUTES": [estimated_minutes],
                "TRIP_MILES_SQ": [trip_miles_sq]
            })

            # Scale the features and prepare the final input dictionary
            scaled_features_array = scaler.transform(features_df[settings.input_features])
            for i, feature_name in enumerate(settings.input_features):
                input_data[feature_name] = np.array([scaled_features_array[0, i]])
                
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
