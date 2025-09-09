# -*- coding: utf-8 -*-
"""Evaluates a trained model on the test dataset."""

import argparse
import json
import pickle
from pathlib import Path

import keras

from linear_regression_taxi_1 import utils, pipeline


# We need the DurationEstimator class here as well for the smart_model
class DurationEstimator:
    def __init__(self, config_path: Path):
        try:
            with open(config_path, 'r') as f:
                self.lookup_table = {float(k): v for k, v in json.load(f).items()}
            self.sorted_upper_bounds = sorted(self.lookup_table.keys())
        except FileNotFoundError:
            print(f"Error: Lookup table not found at {config_path}")
            raise

    def estimate(self, miles: float) -> float:
        for upper_bound in self.sorted_upper_bounds:
            if miles < upper_bound:
                return self.lookup_table[upper_bound]
        return self.lookup_table[self.sorted_upper_bounds[-1]]


def test_model(experiment_name: str, num_worst_predictions: int | None = None):
    """Loads a trained model and evaluates it on the test set."""
    print(f"--- Testing experiment: {experiment_name} ---")

    # --- Setup ---
    # Paths are relative to the project root, where the script should be run from.
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    model_path = artifacts_dir / f"{experiment_name}_model.keras"
    settings_path = artifacts_dir / f"{experiment_name}_settings.pkl"
    large_error_threshold = 5.0

    if not model_path.exists() or not settings_path.exists():
        print(f"Error: Model artifacts for '{experiment_name}' not found. Please run training first.")
        return

    # --- Load artifacts and data ---
    print("Loading model, settings, and test data...")
    model = keras.models.load_model(model_path)
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)
    test_df = pipeline.load_and_prepare_data(
        "artifacts/linear_regression_taxi_1/chicago_taxi_test.csv"
    )

    # --- Feature Engineering (for smart_model specifically) ---
    if experiment_name == "smart_model":
        print("\nSmart model detected. Applying feature engineering to test set...")
        lookup_table_path = artifacts_dir / "time_distance_lookup.json"
        try:
            estimator = DurationEstimator(lookup_table_path)
        except FileNotFoundError:
            print("Could not initialize duration estimator. Aborting.")
            return
        test_df["ESTIMATED_MINUTES"] = test_df["TRIP_MILES"].apply(estimator.estimate)
        print("Feature engineering complete.")

    # --- Prediction and Analysis ---
    print("\nGenerating predictions for the entire test set...")
    all_predictions_df = utils.predict_fare(
        model=model,
        dataset=test_df,
        feature_names=settings.input_features,
        label_name="FARE",
    )

    # Show extreme predictions if requested
    if num_worst_predictions and num_worst_predictions > 0:
        sorted_predictions = all_predictions_df.sort_values(by="L1_LOSS", ascending=False)
        worst_predictions = sorted_predictions.head(num_worst_predictions)
        utils.show_predictions(
            worst_predictions, title=f"TOP {num_worst_predictions} WORST PREDICTIONS (HIGHEST ERROR)"
        )
        best_predictions = sorted_predictions.tail(10).sort_values(by="L1_LOSS", ascending=True)
        utils.show_predictions(
            best_predictions, title="TOP 10 BEST PREDICTIONS (LOWEST ERROR)"
        )

    # --- Final Evaluation and Summary ---
    print("Evaluating model performance on the entire test set...")
    features = {name: test_df[name].values for name in settings.input_features}
    label = test_df["FARE"].values
    results = model.evaluate(features, label, verbose=0)

    large_errors_df = all_predictions_df[all_predictions_df['L1_LOSS'] > large_error_threshold]
    large_error_count = len(large_errors_df)
    total_predictions = len(all_predictions_df)
    large_error_percentage = (large_error_count / total_predictions) * 100 if total_predictions > 0 else 0

    # --- Print Final Report ---
    print(f"\n--- Overall Test Results for: {experiment_name} ---")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print("-" * 35)
    print(f"Errors > ${large_error_threshold:.2f}: {large_error_count} / {total_predictions} ({large_error_percentage:.2f}%)")
    print("-" * 35 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test dataset.")
    parser.add_argument("experiment_name", type=str, help="The name of the experiment to test (e.g., 'single_feature', 'multi_feature', or 'smart_model').")
    parser.add_argument("--show-predictions", type=int, metavar="N", help="Number of worst predictions to display. Also shows top 10 best.")
    args = parser.parse_args()

    test_model(args.experiment_name, args.show_predictions)
