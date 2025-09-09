# -*- coding: utf-8 -*-
"""Handles all data processing for the Rice dataset."""

import pandas as pd

# Import common utilities
from src.common import stats_utils
from src.common.base_data_wrapper import BaseDataWrapper


class RiceDataWrapper(BaseDataWrapper):
    """
    Handles all data loading, preprocessing, and splitting for the Rice dataset
    by inheriting from the BaseDataWrapper.
    """

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects columns, normalizes features, and creates the boolean label."""
        print("\n--- Preprocessing Rice dataset ---")
        
        # 1. Select relevant columns
        relevant_cols = [
            'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
            'Eccentricity', 'Convex_Area', 'Extent', 'Class'
        ]
        processed_df = df[relevant_cols].copy()

        # 2. Normalize numerical features using the common utility
        normalized_df = stats_utils.normalize_features(processed_df)
        print("Numerical features normalized.")

        # 3. Create boolean label
        # Copy the original 'Class' column back as it's needed for the label
        normalized_df['Class'] = processed_df['Class']
        normalized_df['Class_Bool'] = (normalized_df['Class'] == 'Cammeo').astype(int)
        print("Boolean label 'Class_Bool' created.")

        return normalized_df

    def _get_final_columns(self) -> list[str]:
        """Returns the list of columns to be used for modeling."""
        return ['Eccentricity', 'Major_Axis_Length', 'Area', 'Class', 'Class_Bool']

    def _get_dataset_name(self) -> str:
        """Returns the base name for the output CSV files."""
        return "rice"

    def _get_split_args(self) -> dict:
        """
        Returns arguments for a 64/16/20 split.
        (test_size=0.2 means 80% for train+val, val_share_of_train=0.2 of 80% is 16% for val)
        """
        return {"test_size": 0.2, "val_share_of_train": 0.2, "random_state": 42}
