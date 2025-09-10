# -*- coding: utf-8 -*-
"""Evaluates a trained model on the test dataset."""

import argparse
import json
import pickle
from pathlib import Path

import keras

# Import utilities from our common package
from src.common import csv_utils, prediction_utils


def test_model(experiment_name: str, num_worst_predictions: int | None = None):
    """Loads a trained model and evaluates it on the test set."""
    print(f"--- Testing experiment: {experiment_name} ---")

    # --- Setup ---
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    model_path = artifacts_dir / f"{experiment_name}_model.keras"
    settings_path = artifacts_dir / f"{experiment_name}_settings.pkl"
    test_data_path = artifacts_dir / "chicago_taxi_test.csv"
    large_error_threshold = 5.0

    if not model_path.exists() or not settings_path.exists():
        print(f"Error: Model artifacts for '{experiment_name}' not found. Please run training first.")
        return

    # --- Load artifacts and data ---
    print("Loading model, settings, and test data...")
    model = keras.models.load_model(model_path)
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)
    
    # The test data should already be processed, so we just load it.
    test_df = csv_utils.load_csv(str(test_data_path))

    # --- Prediction and Analysis ---
    print("\nGenerating predictions for the entire test set...")
    # Replacing the old function with the correct one from the common utils
    all_predictions_df = prediction_utils.predict_on_dataframe(
        model=model,
        dataset=test_df,
        feature_names=settings.input_features,
        label_name="FARE",
        error_col_name="L1_LOSS" # Explicitly name the error column to match legacy code
    )

    # Show extreme predictions if requested
    if num_worst_predictions and num_worst_predictions > 0:
        sorted_predictions = all_predictions_df.sort_values(by="L1_LOSS", ascending=False)
        worst_predictions = sorted_predictions.head(num_worst_predictions)
        prediction_utils.show_predictions(
            worst_predictions, title=f"TOP {num_worst_predictions} WORST PREDICTIONS (HIGHEST ERROR)"
        )
        best_predictions = sorted_predictions.tail(5).sort_values(by="L1_LOSS", ascending=True)
        prediction_utils.show_predictions(
            best_predictions, title="TOP 5 BEST PREDICTIONS (LOWEST ERROR)"
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
