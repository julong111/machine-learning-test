# -*- coding: utf-8 -*-
"""Common utilities for loading and saving CSV files."""

import os
import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame | None:
    """Loads data from a CSV file.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame with the loaded data, or None if the file is not found.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return None

def save_csv(df: pd.DataFrame, file_path: str):
    """Saves a DataFrame to a CSV file.

    Args:
        df: The pandas DataFrame to save.
        file_path: The full path where the CSV file will be saved.
    """
    # Ensure the directory exists before saving
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving DataFrame to {file_path}...")
    df.to_csv(file_path, index=False)
    print("DataFrame saved successfully.")
