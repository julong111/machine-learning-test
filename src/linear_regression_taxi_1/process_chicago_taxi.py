# -*- coding: utf-8 -*-
import os
import pandas as pd
from split_data import load_data, split_data, save_split_data

def clean_chicago_taxi_data(df):
    """Cleans the Chicago Taxi dataset using tailored, data-driven rules."""
    print("\n--- Starting Data Cleaning Process for Chicago Taxi Data ---")
    initial_rows = len(df)

    # Basic filtering for physical validity
    df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60
    sane_df = df[
        (df["FARE"] > 0) &
        (df["TRIP_MILES"] > 0) &
        (df["TRIP_MINUTES"] > 0)
    ].copy()

    # Configuration for cleaning rules
    short_trip_mile_threshold = 1.0
    short_trip_fare_quantile = 0.999
    average_speed_bounds = (1.0, 80.0)
    lower_bound_price_per_mile = 1.0

    # Separate data for tailored analysis
    short_trips_df = sane_df[sane_df["TRIP_MILES"] <= short_trip_mile_threshold].copy()
    normal_trips_df = sane_df[sane_df["TRIP_MILES"] > short_trip_mile_threshold].copy()

    # Determine fare cap for short trips
    short_trip_fare_cap = short_trips_df["FARE"].quantile(short_trip_fare_quantile)
    print(f"Calculated FARE cap for short trips: ${short_trip_fare_cap:.2f}")

    # Determine price/mile cap for normal trips
    normal_trips_df["price_per_mile"] = normal_trips_df["FARE"] / normal_trips_df["TRIP_MILES"]
    q1 = normal_trips_df["price_per_mile"].quantile(0.25)
    q3 = normal_trips_df["price_per_mile"].quantile(0.75)
    iqr = q3 - q1
    normal_trip_price_cap = q3 + 1.5 * iqr
    print(f"Calculated Price/Mile cap for normal trips: ${normal_trip_price_cap:.2f}")

    # Identify outliers
    short_trip_fare_outliers = short_trips_df[short_trips_df["FARE"] > short_trip_fare_cap].index
    normal_trip_price_outliers = normal_trips_df[normal_trips_df["price_per_mile"] > normal_trip_price_cap].index

    sane_df["price_per_mile"] = sane_df["FARE"] / sane_df["TRIP_MILES"]
    sane_df["average_speed"] = sane_df["TRIP_MILES"] / (sane_df["TRIP_MINUTES"] / 60)
    low_price_indices = sane_df[sane_df["price_per_mile"] < lower_bound_price_per_mile].index
    speed_indices = sane_df[
        (sane_df["average_speed"] < average_speed_bounds[0]) |
        (sane_df["average_speed"] > average_speed_bounds[1])
    ].index

    # Combine and drop outliers
    indices_to_drop = short_trip_fare_outliers.union(normal_trip_price_outliers).union(low_price_indices).union(speed_indices)
    cleaned_df = df.drop(index=indices_to_drop)
    
    # Drop temporary columns
    cleaned_df = cleaned_df.drop(columns=["TRIP_MINUTES", "price_per_mile", "average_speed"], errors='ignore')

    final_rows = len(cleaned_df)
    print(f"Cleaning complete. {initial_rows - final_rows} rows removed, {final_rows} rows remaining.")
    
    return cleaned_df

def main():
    """Main function to run the data processing pipeline."""
    # Determine project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')
    input_csv_path = os.path.join(artifacts_dir, 'chicago_taxi_google.csv')

    # Load data
    raw_df = load_data(input_csv_path)
    if raw_df is None:
        return

    # Clean data
    cleaned_df = clean_chicago_taxi_data(raw_df)

    # Split data
    train_df, val_df, test_df = split_data(cleaned_df)

    # Save data
    save_split_data(train_df, val_df, test_df, artifacts_dir, 'chicago_taxi')
    
    print("\nProcess complete. Your datasets are ready for modeling.")

if __name__ == "__main__":
    main()
