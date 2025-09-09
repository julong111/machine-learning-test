# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(input_csv_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return None

def split_data(df, test_size=0.1, val_share_of_train=1/9, random_state=42):
    """Splits a DataFrame into training, validation, and test sets."""
    print(f"\n--- Splitting data into train/validation/test sets ---")
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_share_of_train, random_state=random_state)
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    return train_df, val_df, test_df

def save_split_data(train_df, val_df, test_df, output_dir, file_prefix):
    """Saves the split datasets to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, f'{file_prefix}_train.csv')
    val_output_path = os.path.join(output_dir, f'{file_prefix}_validation.csv')
    test_output_path = os.path.join(output_dir, f'{file_prefix}_test.csv')

    print(f"\nSaving training set to {train_output_path}...")
    train_df.to_csv(train_output_path, index=False)

    print(f"Saving validation set to {val_output_path}...")
    val_df.to_csv(val_output_path, index=False)

    print(f"Saving test set to {test_output_path}...")
    test_df.to_csv(test_output_path, index=False)
    
    print("\nDatasets saved successfully.")
