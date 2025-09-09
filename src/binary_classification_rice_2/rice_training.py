# -*- coding: utf-8 -*-
"""Main entry point for training binary classification models for the rice dataset."""

import argparse
import os
import sys
import keras
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the refactored classes and utilities
from src.common.types import Experiment, ExperimentSettings
from src.common import csv_utils
from src.common.keras_utils import create_binary_classification_model, prepare_features_and_labels
from src.common.plot_utils import plot_classification_history
from src.binary_classification_rice_2.rice_data_wrapper import RiceDataWrapper

# Set the random seeds for reproducibility
keras.utils.set_random_seed(42)

# Configuration for the model, using our common ExperimentSettings type
MODEL_CONFIGS = {
    "rice_model": ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        input_features=['Eccentricity', 'Major_Axis_Length', 'Area'],
        verbose=1,
    ),
}

def main():
    """Orchestrates the model training process."""
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train a specified classification model.")
    parser.add_argument(
        "model_type",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        default="rice_model",
        nargs='?',
        help=f"The type of model to train. Choose from: {list(MODEL_CONFIGS.keys())}",
    )
    args = parser.parse_args()
    model_type = args.model_type

    # 2. Setup Paths and Configuration
    print(f"\n--- Starting process for model: {model_type} ---")
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    settings = MODEL_CONFIGS[model_type]

    # Define paths for processed data, model, and plots
    train_csv_path = os.path.join(artifacts_dir, "rice_train.csv")
    val_csv_path = os.path.join(artifacts_dir, "rice_validation.csv")
    test_csv_path = os.path.join(artifacts_dir, "rice_test.csv")
    model_path = os.path.join(artifacts_dir, f'{model_type}.keras')
    plot_path = os.path.join(artifacts_dir, f'{model_type}_training_history.png')

    # 3. Conditional Data Loading/Processing
    if all(os.path.exists(p) for p in [train_csv_path, val_csv_path, test_csv_path]):
        print("\nFound pre-processed data. Loading directly from CSV files...")
        train_df = csv_utils.load_csv(train_csv_path)
        val_df = csv_utils.load_csv(val_csv_path)
        test_df = csv_utils.load_csv(test_csv_path)
    else:
        print("\nProcessed data not found. Starting data processing pipeline...")
        input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')
        data_wrapper = RiceDataWrapper(input_csv_path=input_csv_path, artifacts_dir=artifacts_dir)
        try:
            train_df, val_df, test_df = data_wrapper.process_and_split()
        except FileNotFoundError as e:
            print(f"Error during data processing: {e}")
            print("Aborting.")
            return

    if train_df is None or val_df is None or test_df is None:
        print("Error: Data loading failed. Aborting.")
        return

    # 4. Feature and Label Preparation
    label_column = 'Class_Bool'
    train_features, train_labels = prepare_features_and_labels(
        train_df, settings.input_features, label_column
    )
    validation_features, validation_labels = prepare_features_and_labels(
        val_df, settings.input_features, label_column
    )
    test_features, test_labels = prepare_features_and_labels(
        test_df, settings.input_features, label_column
    )

    # 5. Model Training
    model = create_binary_classification_model(
        input_features=settings.input_features,
        learning_rate=settings.learning_rate
    )
    
    history = model.fit(
        x=train_features,
        y=train_labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        verbose=settings.verbose,
        validation_data=(validation_features, validation_labels)
    )
    print("\n--- Model Training Complete ---")

    # 6. Model Evaluation
    print("\n--- Evaluating Model Performance on Test Set ---")
    loss, accuracy, precision, recall, auc = model.evaluate(test_features, test_labels, verbose=settings.verbose)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # 7. Saving Artifacts
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    experiment = Experiment(
        name=model_type,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history)
    )

    plot_classification_history(experiment, save_path=plot_path)
    print(f"Training history plot saved to {plot_path}")

    print(f"\n--- Successfully completed all steps for model: {model_type} ---")

if __name__ == "__main__":
    main()
