# -*- coding: utf-8 -*-
"""Evaluates a trained model on the validation dataset."""

import argparse
import pickle
from pathlib import Path

import keras

# Import pipeline and utils to allow unpickling of the ExperimentSettings object
from src.linear_regression_taxi_1 import pipeline, utils


def validate_model(experiment_name: str):
    """Loads a trained model and evaluates it on the validation set.

    Args:
        experiment_name: The name of the experiment to validate.
    """
    print(f"--- Validating experiment: {experiment_name} ---")

    # Define paths relative to the project root
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    model_path = artifacts_dir / f"{experiment_name}_model.keras"
    settings_path = artifacts_dir / f"{experiment_name}_settings.pkl"

    # Check if model artifacts exist
    if not model_path.exists() or not settings_path.exists():
        print(
            f"Error: Model artifacts for '{experiment_name}' not found. "
            "Please run the training script first."
        )
        return

    # Load model and settings
    print("Loading model and settings...")
    model = keras.models.load_model(model_path)
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)

    # Load and prepare validation data from the correct new path
    print("Loading validation data...")
    validation_df = pipeline.load_and_prepare_data(
        "artifacts/linear_regression_taxi_1/chicago_taxi_validation.csv"
    )

    # Prepare features and label for evaluation
    # The feature engineering for the 'smart_model' is not applied here.
    # This validation script is for the simpler models.
    features = {name: validation_df[name].values for name in settings.input_features}
    label = validation_df["FARE"].values

    # Evaluate the model
    print("Evaluating model performance on the validation set...")
    results = model.evaluate(features, label, verbose=0)

    # Print the results
    print("\n--- Validation Results ---")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a trained model on the validation dataset."
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="The name of the experiment to validate (e.g., 'single_feature').",
    )
    args = parser.parse_args()

    validate_model(args.experiment_name)
