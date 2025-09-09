# -*- coding: utf-8 -*-
"""One-time script to perform all data preparation for the rice dataset."""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.binary_classification_rice_2.rice_data_wrapper import RiceDataWrapper

def main():
    """Orchestrates the data preparation process."""
    print("\n--- Starting Data Preparation for Rice Dataset ---")

    # 1. Setup Paths
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')

    # 2. Run Data Processing
    # This will load raw data, clean it, split it,
    # and save the final train/val/test CSVs to the artifacts directory.
    data_wrapper = RiceDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
    try:
        data_wrapper.process_and_split()
    except FileNotFoundError as e:
        print(f"Error during data processing: {e}")
        print("Aborting.")
        return
    
    print("\n--- Data Preparation Complete ---")
    print(f"Clean datasets are now available in: {artifacts_dir}")

if __name__ == "__main__":
    main()
