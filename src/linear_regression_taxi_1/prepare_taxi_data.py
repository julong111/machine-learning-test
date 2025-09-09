# -*- coding: utf-8 -*-
"""One-time script to perform all data preparation for the taxi dataset."""

import os

from src.linear_regression_taxi_1.taxi_data_wrapper import TaxiDataWrapper

def main():
    """Orchestrates the data preparation process."""
    print("\n--- Starting Data Preparation for Chicago Taxi Dataset ---")

    # 1. Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')
    input_csv_path = os.path.join(artifacts_dir, 'chicago_taxi_google.csv')

    # 2. Run Data Processing
    # This will load raw data, clean it, engineer features, split it,
    # and save the final train/val/test CSVs to the artifacts directory.
    data_wrapper = TaxiDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
    try:
        data_wrapper.process_and_split()
    except FileNotFoundError as e:
        print(f"Error during data processing: {e}")
        print("Aborting.")
        return
    
    print("\n--- Data Preparation Complete ---")
    print(f"Clean, feature-engineered datasets are now available in: {artifacts_dir}")

if __name__ == "__main__":
    main()
