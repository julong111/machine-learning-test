# -*- coding: utf-8 -*-
import os
import pandas as pd
import pytest

# Define paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# This is the directory where the new, refactored script saves its files.
actual_output_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')

# Before running this test, you must manually run the *original* script 
# and ensure it saves its output to the directory below.
expected_output_dir = os.path.join(project_root, 'artifacts', 'expected_output')

# List of files to compare
file_names = [
    'chicago_taxi_train.csv',
    'chicago_taxi_validation.csv',
    'chicago_taxi_test.csv'
]

@pytest.mark.parametrize("file_name", file_names)
def test_csv_outputs_are_identical(file_name):
    """Compares a single CSV file from the actual and expected output directories."""
    print(f"\nComparing {file_name}...")
    
    actual_path = os.path.join(actual_output_dir, file_name)
    expected_path = os.path.join(expected_output_dir, file_name)

    # Check if files exist
    assert os.path.exists(actual_path), f"Actual output file not found: {actual_path}"
    assert os.path.exists(expected_path), f"Expected output file not found: {expected_path}"

    # Read CSVs into pandas DataFrames
    actual_df = pd.read_csv(actual_path)
    expected_df = pd.read_csv(expected_path)

    # Use pandas testing utility to compare DataFrames
    # This checks for equality in data, columns, index, and dtypes.
    try:
        pd.testing.assert_frame_equal(actual_df, expected_df)
        print(f"✅ {file_name} is identical.")
    except AssertionError as e:
        print(f"❌ {file_name} is different.")
        # Re-raise the exception to make the pytest test fail
        raise e

