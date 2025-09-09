# -*- coding: utf-8 -*-
"""Analyzes the relationship between trip distance and duration and saves the result."""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_and_save_lookup():
    """Loads clean data, calculates average duration for distance bins, and saves to JSON."""
    # --- Configuration ---
    # Paths are relative to the project root, where the script should be run from.
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    input_csv_path = artifacts_dir / "chicago_taxi_train.csv"
    output_json_path = artifacts_dir / "time_distance_lookup.json"

    # Ensure the artifacts directory exists
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    distance_bins = list(range(0, 31)) + [np.inf]

    # --- Data Loading ---
    print(f"Loading clean training data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # --- Analysis ---
    print("Analyzing relationship between trip distance and duration...")
    df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60
    df["distance_bin"] = pd.cut(df["TRIP_MILES"], bins=distance_bins, right=False)

    # Create the lookup table
    time_distance_lookup = df.groupby("distance_bin")["TRIP_MINUTES"].mean().reset_index()
    time_distance_lookup.rename(columns={"TRIP_MINUTES": "average_minutes"}, inplace=True)

    # --- Convert the lookup table to a simple dictionary for JSON ---
    # The format will be { "upper_bound_of_bin": average_minutes }
    # e.g., { "1.0": 5.83, "2.0": 8.21, ... }
    lookup_dict = {}
    for _, row in time_distance_lookup.iterrows():
        # The interval's right side is the upper bound
        upper_bound = row["distance_bin"].right
        lookup_dict[str(upper_bound)] = round(row["average_minutes"], 2)

    # --- Save to JSON ---
    print(f"Saving lookup table to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(lookup_dict, f, indent=4)

    print("\n--- Distance-to-Average-Time Lookup Table ---")
    print(time_distance_lookup.to_string(index=False))
    print(f"\nSuccessfully created and saved the lookup table to '{output_json_path}'.")


if __name__ == "__main__":
    analyze_and_save_lookup()
