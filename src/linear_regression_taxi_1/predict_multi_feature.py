# -*- coding: utf-8 -*-
"""Script to make predictions using the multi-feature model."""

import pickle
from pathlib import Path

import keras
import numpy as np

# Import utils to allow unpickling of the ExperimentSettings object.
from src.linear_regression_taxi_1 import utils

# --- Configuration ---
EXPERIMENT_NAME = "multi_feature"
# The script is run from the project root, so the path is relative to that.
ARTIFACTS_DIR = Path("artifacts") / "linear_regression_taxi_1"
# -------------------

# Construct paths to the model and settings files
model_path = ARTIFACTS_DIR / f"{EXPERIMENT_NAME}_model.keras"
settings_path = ARTIFACTS_DIR / f"{EXPERIMENT_NAME}_settings.pkl"

# Load the trained model and its settings
try:
    model = keras.models.load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)
    print(f"Settings '{settings_path}' loaded successfully.")
except Exception as e:
    print(f"Error loading model or settings: {e}")
    print(
        "Please ensure you have run the corresponding training script "
        f"('train_{EXPERIMENT_NAME}.py') first."
    )
    exit()


def predict_fare(trip_miles: float, trip_minutes: float) -> float:
    """Predicts taxi fare based on trip miles and minutes."""
    input_data = {
        "TRIP_MILES": np.array([trip_miles]),
        "TRIP_MINUTES": np.array([trip_minutes]),
    }
    prediction = model.predict(input_data, verbose=0)
    return prediction[0][0]


if __name__ == "__main__":
    example_trip_miles = 5.0
    example_trip_minutes = 10.0
    predicted_fare = predict_fare(
        trip_miles=example_trip_miles, trip_minutes=example_trip_minutes
    )

    print("\n--- Prediction Example ---")
    print(f"Trip distance: {example_trip_miles} miles")
    print(f"Trip duration: {example_trip_minutes} minutes")
    print(f"Predicted fare: ${predicted_fare:.2f}")
    print("------------------------")
