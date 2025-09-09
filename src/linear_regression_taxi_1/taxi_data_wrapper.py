# -*- coding: utf-8 -*-
"""Handles all data processing for the Chicago Taxi dataset."""

import os
import json
import numpy as np
import pandas as pd

# Import our new common utilities
from src.common import csv_utils
from src.common import split_utils
from src.common import stats_utils

class TaxiDataWrapper:
    """
    Handles all data loading, cleaning, feature engineering, and splitting
    for the Chicago Taxi dataset.
    """

    def __init__(self, input_csv_path: str, artifacts_dir: str):
        """Initializes the data wrapper."""
        self.input_csv_path = input_csv_path
        self.artifacts_dir = artifacts_dir
        self.time_distance_lookup = {}
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _apply_cleaning_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the full set of cleaning rules to the taxi dataset."""
        print("\n--- Starting Data Cleaning Process for Chicago Taxi Data ---")
        initial_rows = len(df)

        df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60
        sane_df = df[(df["FARE"] > 0) & (df["TRIP_MILES"] > 0) & (df["TRIP_MINUTES"] > 0)].copy()

        short_trip_mile_threshold = 1.0
        short_trip_fare_quantile = 0.999
        average_speed_bounds = (1.0, 80.0)
        lower_bound_price_per_mile = 1.0

        short_trips_df = sane_df[sane_df["TRIP_MILES"] <= short_trip_mile_threshold].copy()
        # Corrected the typo in the line below
        normal_trips_df = sane_df[sane_df["TRIP_MILES"] > short_trip_mile_threshold].copy()

        short_trip_fare_cap = short_trips_df["FARE"].quantile(short_trip_fare_quantile)
        print(f"Calculated FARE cap for short trips: ${short_trip_fare_cap:.2f}")
        short_trip_fare_outliers = short_trips_df[short_trips_df["FARE"] > short_trip_fare_cap].index

        normal_trips_df["price_per_mile"] = normal_trips_df["FARE"] / normal_trips_df["TRIP_MILES"]
        _, normal_trip_price_cap = stats_utils.get_iqr_bounds(normal_trips_df["price_per_mile"])
        normal_trip_price_outliers = normal_trips_df[normal_trips_df["price_per_mile"] > normal_trip_price_cap].index

        sane_df["price_per_mile"] = sane_df["FARE"] / sane_df["TRIP_MILES"]
        sane_df["average_speed"] = sane_df["TRIP_MILES"] / (sane_df["TRIP_MINUTES"] / 60)
        low_price_indices = sane_df[sane_df["price_per_mile"] < lower_bound_price_per_mile].index
        speed_indices = sane_df[(sane_df["average_speed"] < average_speed_bounds[0]) | (sane_df["average_speed"] > average_speed_bounds[1])].index

        indices_to_drop = short_trip_fare_outliers.union(normal_trip_price_outliers).union(low_price_indices).union(speed_indices)
        cleaned_df = df.drop(index=indices_to_drop)

        final_rows = len(cleaned_df)
        print(f"Cleaning complete. {initial_rows - final_rows} rows removed, {final_rows} rows remaining.")
        return cleaned_df

    def _create_time_distance_lookup(self, df: pd.DataFrame):
        """Analyzes trip distance and duration and saves the result."""
        print("\n--- Analyzing relationship between trip distance and duration ---")
        output_json_path = os.path.join(self.artifacts_dir, "time_distance_lookup.json")
        
        if "TRIP_MINUTES" not in df.columns:
            df["TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60

        distance_bins = list(range(0, 31)) + [np.inf]
        df["distance_bin"] = pd.cut(df["TRIP_MILES"], bins=distance_bins, right=False)

        lookup_df = df.groupby("distance_bin")["TRIP_MINUTES"].mean().reset_index()
        
        lookup_dict = {str(row["distance_bin"].right): round(row["TRIP_MINUTES"], 2) for _, row in lookup_df.iterrows()}
        self.time_distance_lookup = {float(k): v for k, v in lookup_dict.items()}

        print(f"Saving lookup table to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(lookup_dict, f, indent=4)
        print("Lookup table created and saved successfully.")

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features for the model."""
        print("\n--- Engineering features ---")
        df_engineered = df.copy()

        if "TRIP_MINUTES" not in df_engineered.columns:
            df_engineered["TRIP_MINUTES"] = df_engineered["TRIP_SECONDS"] / 60
        
        if not self.time_distance_lookup:
            print("Warning: Time-distance lookup table is empty. Cannot create ESTIMATED_MINUTES.")
            return df_engineered

        sorted_upper_bounds = sorted(self.time_distance_lookup.keys())
        
        def estimate(miles: float) -> float:
            for upper_bound in sorted_upper_bounds:
                if miles < upper_bound:
                    return self.time_distance_lookup[upper_bound]
            return self.time_distance_lookup[sorted_upper_bounds[-1]]

        df_engineered["ESTIMATED_MINUTES"] = df_engineered["TRIP_MILES"].apply(estimate)
        print("Engineered 'TRIP_MINUTES' and 'ESTIMATED_MINUTES' features.")
        return df_engineered

    def process_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates the full data processing pipeline."""
        raw_df = csv_utils.load_csv(self.input_csv_path)
        if raw_df is None:
            raise FileNotFoundError(f"Raw data not found at {self.input_csv_path}")

        cleaned_df = self._apply_cleaning_rules(raw_df)

        train_df, val_df, test_df = split_utils.split_dataframe(cleaned_df)

        self._create_time_distance_lookup(train_df)

        train_df = self._feature_engineer(train_df)
        val_df = self._feature_engineer(val_df)
        test_df = self._feature_engineer(test_df)

        # Select only the columns needed for modeling to create a clean final dataset
        final_columns = ['TRIP_MILES', 'TRIP_MINUTES', 'ESTIMATED_MINUTES', 'FARE']
        
        # Ensure all required columns are present before selection
        for df in [train_df, val_df, test_df]:
            for col in final_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in the DataFrame after processing.")

        train_df_final = train_df[final_columns]
        val_df_final = val_df[final_columns]
        test_df_final = test_df[final_columns]
        
        print(f"\nFinal datasets will contain only the following columns: {final_columns}")

        # Save the final, streamlined datasets
        csv_utils.save_csv(train_df_final, os.path.join(self.artifacts_dir, "chicago_taxi_train.csv"))
        csv_utils.save_csv(val_df_final, os.path.join(self.artifacts_dir, "chicago_taxi_validation.csv"))
        csv_utils.save_csv(test_df_final, os.path.join(self.artifacts_dir, "chicago_taxi_test.csv"))
        
        print("\nProcess complete. Your datasets are ready for modeling.")
        return train_df_final, val_df_final, test_df_final
