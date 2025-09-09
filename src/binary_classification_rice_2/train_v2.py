# -*- coding: utf-8 -*-
# train_v2.py

import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the random seeds for reproducibility
keras.utils.set_random_seed(42)

def load_and_preprocess_data(input_path):
    """Loads, preprocesses, and cleans the rice dataset."""
    print(f"Loading and preprocessing data from {input_path}...")
    
    rice_dataset_raw = pd.read_csv(input_path)

    # Select relevant columns
    rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]]

    # Normalize numerical features
    feature_mean = rice_dataset.mean(numeric_only=True)
    feature_std = rice_dataset.std(numeric_only=True)
    numerical_features = rice_dataset.select_dtypes('number').columns
    normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

    # Copy the class to the new dataframe and create boolean label
    normalized_dataset['Class'] = rice_dataset['Class']
    normalized_dataset['Class_Bool'] = (normalized_dataset['Class'] == 'Cammeo').astype(int)
    
    print("Data loading and preprocessing complete.")
    return normalized_dataset

def create_model(input_features, learning_rate):
    """Create and compile a simple binary classification model."""
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in input_features]
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(units=1, name='dense_layer', activation=keras.activations.sigmoid)(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            keras.metrics.Precision(name='precision', thresholds=0.5),
            keras.metrics.Recall(name='recall', thresholds=0.5),
            keras.metrics.AUC(num_thresholds=100, name='auc'),
        ],
    )
    return model

def plot_and_save_history(history, artifacts_dir):
    """Plots the training and validation metrics and saves the plot."""
    history_df = pd.DataFrame(history.history)
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training History', fontsize=16)

    # Plot Loss
    axes[0, 0].plot(history_df['loss'], label='Training Loss')
    axes[0, 0].plot(history_df['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss vs. Epochs')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot Accuracy
    axes[0, 1].plot(history_df['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history_df['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy vs. Epochs')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Plot Precision & Recall
    axes[1, 0].plot(history_df['precision'], label='Training Precision')
    axes[1, 0].plot(history_df['val_precision'], label='Validation Precision')
    axes[1, 0].plot(history_df['recall'], label='Training Recall', linestyle='--')
    axes[1, 0].plot(history_df['val_recall'], label='Validation Recall', linestyle='--')
    axes[1, 0].set_title('Precision & Recall vs. Epochs')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

    # Plot AUC
    axes[1, 1].plot(history_df['auc'], label='Training AUC')
    axes[1, 1].plot(history_df['val_auc'], label='Validation AUC')
    axes[1, 1].set_title('AUC vs. Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plot_path = os.path.join(artifacts_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"\nTraining history plot saved to {plot_path}")

def main():
    """Main function to orchestrate the model training process."""
    # --- Configuration ---
    LEARNING_RATE = 0.001
    NUMBER_EPOCHS = 60
    BATCH_SIZE = 100
    INPUT_FEATURES = ['Eccentricity', 'Major_Axis_Length', 'Area']

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
    model_path = os.path.join(artifacts_dir, 'rice_model.keras')
    input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')
    
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Data Loading & Splitting ---
    processed_data = load_and_preprocess_data(input_csv_path)

    # Split data into 80% train/validation and 20% test
    train_val_df, test_df = train_test_split(processed_data, test_size=0.2, random_state=42)
    # Split the 80% into training and validation (80/20 split, which is 64/16 of the total)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    print(f"\nData split complete:")
    print(f"- Training set: {len(train_df)} rows")
    print(f"- Validation set: {len(val_df)} rows")
    print(f"- Test set: {len(test_df)} rows")

    # --- Feature and Label Preparation ---
    label_column = 'Class_Bool'

    train_labels = train_df[label_column].to_numpy()
    train_features = {name: np.array(train_df[name]) for name in INPUT_FEATURES}
    
    validation_labels = val_df[label_column].to_numpy()
    validation_features = {name: np.array(val_df[name]) for name in INPUT_FEATURES}

    # --- Model Training ---
    print("\n--- Starting Model Training ---")
    model = create_model(INPUT_FEATURES, LEARNING_RATE)
    
    history = model.fit(
        x=train_features,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUMBER_EPOCHS,
        verbose=1,
        validation_data=(validation_features, validation_labels)
    )
    print("--- Model Training Complete ---")

    # --- Saving Artifacts ---
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    plot_and_save_history(history, artifacts_dir)

if __name__ == "__main__":
    main()
