# -*- coding: utf-8 -*-
"""Cleans the raw dataset using tailored, data-driven rules and splits it."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_and_split_data():
    """Applies the final, most advanced cleaning logic and splits the data."""
    # --- Configuration ---
    # Correctly determine the project root, which is two levels up from the script's location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')
    input_csv_path = os.path.join(artifacts_dir, 'chicago_taxi_google.csv')

    # Define cleaning rule parameters
    short_trip_mile_threshold = 1.0
    short_trip_fare_quantile = 0.999  # Use 99.9th percentile for short trip fare cap
    average_speed_bounds = (1.0, 80.0)
    lower_bound_price_per_mile = 1.0

    # --- 1. Data Loading ---
    print(f"Loading raw data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
        initial_rows = len(df)
        print(f"Successfully loaded {initial_rows} raw records.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # --- 2. Data Cleaning ---
    print("\n--- Starting Data Cleaning Process (v3.0) ---")

    # Basic filtering for physical validity
    df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60
    sane_df = df[
        (df["FARE"] > 0) &
        (df["TRIP_MILES"] > 0) &
        (df["TRIP_MINUTES"] > 0)
    ].copy()

    # --- Separate data into two groups for tailored analysis ---
    short_trips_df = sane_df[sane_df["TRIP_MILES"] <= short_trip_mile_threshold].copy()
    normal_trips_df = sane_df[sane_df["TRIP_MILES"] > short_trip_mile_threshold].copy()

    # --- Determine thresholds for each group ---
    # For short trips, find the fare cap using percentiles
    short_trip_fare_cap = short_trips_df["FARE"].quantile(short_trip_fare_quantile)
    print(f"Calculated FARE cap for short trips (<={short_trip_mile_threshold} miles) using {short_trip_fare_quantile*100}th percentile: ${short_trip_fare_cap:.2f}")

    # For normal trips, find the price/mile cap using IQR
    normal_trips_df["price_per_mile"] = normal_trips_df["FARE"] / normal_trips_df["TRIP_MILES"]
    q1 = normal_trips_df["price_per_mile"].quantile(0.25)
    q3 = normal_trips_df["price_per_mile"].quantile(0.75)
    iqr = q3 - q1
    normal_trip_price_cap = q3 + 1.5 * iqr
    print(f"Calculated Price/Mile cap for normal trips (>{short_trip_mile_threshold} miles) using IQR method: ${normal_trip_price_cap:.2f}")

    # --- Identify all outliers based on the tailored rules ---
    # Rule 1: Short trips with excessively high fare
    short_trip_fare_outliers = short_trips_df[short_trips_df["FARE"] > short_trip_fare_cap].index

    # Rule 2: Normal trips with excessively high price/mile
    normal_trip_price_outliers = normal_trips_df[normal_trips_df["price_per_mile"] > normal_trip_price_cap].index

    # Rule 3 & 4: Universal rules for all trips (low price/mile and bad speed)
    sane_df["price_per_mile"] = sane_df["FARE"] / sane_df["TRIP_MILES"]
    sane_df["average_speed"] = sane_df["TRIP_MILES"] / (sane_df["TRIP_MINUTES"] / 60)
    low_price_indices = sane_df[sane_df["price_per_mile"] < lower_bound_price_per_mile].index
    speed_indices = sane_df[
        (sane_df["average_speed"] < average_speed_bounds[0]) |
        (sane_df["average_speed"] > average_speed_bounds[1])
    ].index

    # Combine all unique indices to be dropped
    indices_to_drop = short_trip_fare_outliers.union(normal_trip_price_outliers).union(low_price_indices).union(speed_indices)
    print(f"\nIdentified {len(indices_to_drop)} total rows to remove based on all rules.")

    # Perform the actual drop
    cleaned_df = df.drop(index=indices_to_drop)
    final_rows = len(cleaned_df)
    print(f"Cleaning complete. {initial_rows - final_rows} rows removed, {final_rows} rows remaining.")

    # --- 3. Data Splitting ---
    print("\n--- Splitting cleaned data into 80/10/10 sets ---")
    cleaned_df = cleaned_df.drop(columns=["TRIP_MINUTES"])
    train_val_df, test_df = train_test_split(cleaned_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=1/9, random_state=42)

    # --- 4. Saving Files ---
    os.makedirs(artifacts_dir, exist_ok=True) # Ensure the directory exists
    train_output_path = os.path.join(artifacts_dir, 'chicago_taxi_train.csv')
    val_output_path = os.path.join(artifacts_dir, 'chicago_taxi_validation.csv')
    test_output_path = os.path.join(artifacts_dir, 'chicago_taxi_test.csv')

    print(f"\nSaving training set to {train_output_path} ({len(train_df)} rows)... Overwriting previous file.")
    train_df.to_csv(train_output_path, index=False)

    print(f"Saving validation set to {val_output_path} ({len(val_df)} rows)... Overwriting previous file.")
    val_df.to_csv(val_output_path, index=False)

    print(f"Saving test set to {test_output_path} ({len(test_df)} rows)... Overwriting previous file.")
    test_df.to_csv(test_output_path, index=False)

    print("\nData cleaning and splitting process complete. Your datasets are now ready for modeling.")


if __name__ == "__main__":
    clean_and_split_data()
