# -*- coding: utf-8 -*-
"""Defines the abstract base class for all data wrappers."""

import os
import pandas as pd
from abc import ABC, abstractmethod

from src.common import csv_utils
from src.common import split_utils

class BaseDataWrapper(ABC):
    """
    Abstract base class for a data wrapper.

    This class defines a standard pipeline for processing data:
    1. Load raw data from a CSV file.
    2. Perform dataset-specific preprocessing.
    3. Split the data into training, validation, and test sets.
    4. Perform optional feature engineering after the split.
    5. Select final columns for the model.
    6. Save the processed datasets to disk.

    Subclasses must implement the abstract methods to provide
    dataset-specific logic.
    """

    def __init__(self, input_csv_path: str, artifacts_dir: str):
        """Initializes the data wrapper."""
        self.input_csv_path = input_csv_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @abstractmethod
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs all dataset-specific cleaning and preprocessing."""
        pass

    @abstractmethod
    def _get_final_columns(self) -> list[str]:
        """Returns the list of columns to keep for the final datasets."""
        pass

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Returns the base name for the output CSV files (e.g., 'rice')."""
        pass

    def _get_split_args(self) -> dict:
        """
        Returns a dictionary of arguments for the split_dataframe utility.
        Subclasses can override this to customize the data split.
        """
        # Corresponds to the default 80/10/10 split in split_utils
        return {}

    def _post_split_processing(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Hook for optional processing after the data split (e.g., stateful
        feature engineering where logic is learned from the training set
        and applied to all sets). Base implementation does nothing.
        """
        return train_df, val_df, test_df

    def process_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates the full data processing pipeline."""
        # 1. Load
        print(f"\n--- Starting data processing for {self._get_dataset_name()} dataset ---")
        raw_df = csv_utils.load_csv(self.input_csv_path)
        if raw_df is None:
            raise FileNotFoundError(f"Raw data not found at {self.input_csv_path}")

        # 2. Preprocess
        preprocessed_df = self._preprocess(raw_df)

        # 3. Split
        split_args = self._get_split_args()
        train_df, val_df, test_df = split_utils.split_dataframe(preprocessed_df, **split_args)

        # 4. Post-split processing (e.g., feature engineering)
        train_df, val_df, test_df = self._post_split_processing(train_df, val_df, test_df)

        # 5. Select final columns
        final_columns = self._get_final_columns()

        # Ensure all required columns are present before selection
        for df in [train_df, val_df, test_df]:
            for col in final_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in the DataFrame after processing.")

        train_df_final = train_df[final_columns]
        val_df_final = val_df[final_columns]
        test_df_final = test_df[final_columns]
        
        print(f"\nFinal datasets will contain only the following columns: {final_columns}")

        # 6. Save
        dataset_name = self._get_dataset_name()
        csv_utils.save_csv(train_df_final, os.path.join(self.artifacts_dir, f"{dataset_name}_train.csv"))
        csv_utils.save_csv(val_df_final, os.path.join(self.artifacts_dir, f"{dataset_name}_validation.csv"))
        csv_utils.save_csv(test_df_final, os.path.join(self.artifacts_dir, f"{dataset_name}_test.csv"))
        
        print(f"\nProcess complete. {dataset_name.replace('_', ' ')} datasets are ready for modeling.")
        return train_df_final, val_df_final, test_df_final