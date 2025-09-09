# -*- coding: utf-8 -*-
"""Common utility for splitting a DataFrame."""

import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataframe(df: pd.DataFrame, test_size=0.1, val_share_of_train=1/9, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into training, validation, and test sets (e.g., 80/10/10).

    Args:
        df: The DataFrame to split.
        test_size: The proportion of the dataset to allocate to the test set.
        val_share_of_train: The proportion of the remaining data to allocate to the validation set.
                           The default 1/9 of the (1-test_size) data results in a 10% validation set
                           from the original total. (e.g., 1/9 * 0.9 = 0.1)
        random_state: The seed for the random number generator for reproducibility.

    Returns:
        A tuple containing the training, validation, and test DataFrames.
    """
    print(f"\n--- Splitting data into train/validation/test sets ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    # val_share_of_train is the proportion of the train_val_df, not the original df
    train_df, val_df = train_test_split(train_val_df, test_size=val_share_of_train, random_state=random_state)

    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    return train_df, val_df, test_df
