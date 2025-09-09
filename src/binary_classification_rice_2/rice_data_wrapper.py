# -*- coding: utf-8 -*-
"""Handles all data processing for the Rice dataset."""

import os
import pandas as pd

from src.common import csv_utils
from src.common import split_utils

class RiceDataWrapper:
    """
    Handles all data loading, preprocessing, and splitting for the Rice dataset.
    """

    def __init__(self, input_csv_path: str, artifacts_dir: str):
        """Initializes the data wrapper."""
        self.input_csv_path = input_csv_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects columns, normalizes features, and creates the boolean label."""
        print("\n--- Preprocessing Rice dataset ---")
        
        # 1. Select relevant columns
        relevant_cols = [
            'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
            'Eccentricity', 'Convex_Area', 'Extent', 'Class'
        ]
        processed_df = df[relevant_cols].copy()

        # 2. Normalize numerical features (Z-score normalization)
        numerical_features = processed_df.select_dtypes('number').columns
        feature_mean = processed_df[numerical_features].mean()
        feature_std = processed_df[numerical_features].std()
        processed_df[numerical_features] = (processed_df[numerical_features] - feature_mean) / feature_std
        print("Numerical features normalized.")

        # 3. Create boolean label
        processed_df['Class_Bool'] = (processed_df['Class'] == 'Cammeo').astype(int)
        print("Boolean label 'Class_Bool' created.")

        return processed_df

    def process_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates the full data processing pipeline for the Rice dataset."""
        raw_df = csv_utils.load_csv(self.input_csv_path)
        if raw_df is None:
            raise FileNotFoundError(f"Raw data not found at {self.input_csv_path}")

        preprocessed_df = self._preprocess(raw_df)

        # Use a 64/16/20 split, which corresponds to test_size=0.2 and val_share_of_train=0.2
        train_df, val_df, test_df = split_utils.split_dataframe(
            preprocessed_df, test_size=0.2, val_share_of_train=0.2, random_state=42
        )

        # Save the processed and split datasets
        csv_utils.save_csv(train_df, os.path.join(self.artifacts_dir, "rice_train.csv"))
        csv_utils.save_csv(val_df, os.path.join(self.artifacts_dir, "rice_validation.csv"))
        csv_utils.save_csv(test_df, os.path.join(self.artifacts_dir, "rice_test.csv"))
        
        print("\nProcess complete. Rice datasets are ready for modeling.")
        return train_df, val_df, test_df
