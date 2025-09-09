# -*- coding: utf-8 -*-
"""Analyze and find outlier records using intelligent, compound rules."""

import os
import pandas as pd


def find_and_analyze_outliers():
    """Loads data, determines thresholds, and uses compound rules to find outliers."""
    # --- Configuration ---
    # Set pandas display options to show all rows without truncation
    pd.set_option('display.max_rows', None)

    # Define the final display order and sorting criteria
    columns_to_display = ["TRIP_MINUTES", "price_per_mile", "TRIP_MILES", "FARE", "TIPS", "TIPS_RATE", "average_speed"]
    sort_criteria = ["price_per_mile", "TRIP_MILES", "FARE", "TIPS"]

    average_speed_bounds = (1.0, 80.0)   # (mph)
    short_trip_threshold = 1.0           # (miles) - trips below this are considered short/base fare

    # --- Data Loading ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Updated to use the new, full dataset
    input_csv_path = os.path.join(project_root, 'chicago_taxi_google.csv')

    print(f"Loading data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # --- Pre-computation and Base Filtering ---
    df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60
    sane_df = df[
        (df["FARE"] > 0) &
        (df["TRIP_MILES"] > 0) &
        (df["TRIP_MINUTES"] > 0)
    ].copy()
    sane_df["price_per_mile"] = sane_df["FARE"] / sane_df["TRIP_MILES"]
    sane_df["average_speed"] = sane_df["TRIP_MILES"] / (sane_df["TRIP_MINUTES"] / 60)
    sane_df["TIPS_RATE"] = (sane_df["TIPS"] / sane_df["FARE"]).fillna(0)

    print(f"\n--- Analyzing {len(sane_df)} potentially valid records ---\n")

    # --- Step 1 & 2: Determine Price/Mile Threshold using IQR ---
    print("--- Analyzing 'price_per_mile' distribution to determine a data-driven upper bound ---")
    price_stats = sane_df["price_per_mile"].describe()
    print(price_stats)
    print("-" * 80)

    q1 = price_stats['25%']
    q3 = price_stats['75%']
    iqr = q3 - q1
    upper_bound_price = q3 + 1.5 * iqr
    lower_bound_price = 1.0

    print(f"IQR = {iqr:.2f}, Calculated Upper Bound (Q3 + 1.5*IQR) = {upper_bound_price:.2f}\n")

    # --- Step 3: Find Outliers using Compound and Relational Rules ---
    all_outlier_indices = set()

    # Rule 1: Price per mile is too low
    low_price_outliers = sane_df[sane_df["price_per_mile"] < lower_bound_price]
    if not low_price_outliers.empty:
        print(f"--- Found {len(low_price_outliers)} outliers: Price/Mile < ${lower_bound_price:.2f} ---")
        print(low_price_outliers[columns_to_display].sort_values(by=sort_criteria))
        all_outlier_indices.update(low_price_outliers.index)
        print("-" * 80 + "\n")

    # Rule 2: COMPOUND RULE for high price per mile
    high_price_outliers = sane_df[
        (sane_df["price_per_mile"] > upper_bound_price) & 
        (sane_df["TRIP_MILES"] > short_trip_threshold)
    ]
    if not high_price_outliers.empty:
        print(f"--- Found {len(high_price_outliers)} outliers: High Price/Mile on Non-Short Trips ---")
        print(f"(Price/Mile > ${upper_bound_price:.2f} AND Trip Miles > {short_trip_threshold})")
        print(high_price_outliers[columns_to_display].sort_values(by=sort_criteria))
        all_outlier_indices.update(high_price_outliers.index)
        print("-" * 80 + "\n")

    # Rule 3: Unreasonable average speed
    speed_outliers_df = sane_df[
        (sane_df["average_speed"] < average_speed_bounds[0]) |
        (sane_df["average_speed"] > average_speed_bounds[1])
    ]
    if not speed_outliers_df.empty:
        print(f"--- Found {len(speed_outliers_df)} outliers: Unreasonable Average Speed (Outside {average_speed_bounds} mph) ---")
        print(speed_outliers_df[columns_to_display].sort_values(by="average_speed"))
        all_outlier_indices.update(speed_outliers_df.index)
        print("-" * 80 + "\n")

    # --- Summary ---
    print("--- Final Summary ---")
    print(f"Total unique outlier rows to be removed: {len(all_outlier_indices)}")


if __name__ == "__main__":
    find_and_analyze_outliers()
