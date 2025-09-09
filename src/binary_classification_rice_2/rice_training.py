# -*- coding: utf-8 -*-
"""Main entry point for training the logistic regression model for the Rice dataset."""

import os
import keras

# Import the new refactored classes and types
from src.common.types import ExperimentSettings
from src.binary_classification_rice_2.rice_data_wrapper import RiceDataWrapper
from src.binary_classification_rice_2.logistic_regression_trainer import LogisticRegressionTrainer

def main():
    """Orchestrates the model training process for the Rice dataset."""
    # Set the random seeds for reproducibility
    keras.utils.set_random_seed(42)

    # 1. Setup Paths and Configuration
    print("\n--- Starting process for Rice Classification Model ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')
    
    # Define the experiment settings
    settings = ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        input_features=['Eccentricity', 'Major_Axis_Length', 'Area'],
        verbose=1,
    )
    experiment_name = "rice_logistic_regression"

    # 2. Data Processing
    # The wrapper handles all data loading, preprocessing, and splitting
    data_wrapper = RiceDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
    try:
        train_df, val_df, test_df = data_wrapper.process_and_split()
    except FileNotFoundError as e:
        print(f"Error during data processing: {e}")
        print("Aborting.")
        return

    # 3. Model Training
    # The trainer handles all model-related tasks
    trainer = LogisticRegressionTrainer(experiment_name=experiment_name, settings=settings)
    trainer.build_model()
    trainer.train(train_df, val_df)
    trainer.evaluate(test_df)
    trainer.save(artifacts_dir)

    print(f"\n--- Successfully completed all steps for model: {experiment_name} ---")

if __name__ == "__main__":
    main()
