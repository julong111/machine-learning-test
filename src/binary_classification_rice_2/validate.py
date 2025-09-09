# validate.py

import argparse
import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd

# Set the random seeds
keras.utils.set_random_seed(42)

# Load the dataset
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

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

# Normalize data
feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

# Copy the class to the new dataframe
normalized_dataset['Class'] = rice_dataset['Class']

# Label data
normalized_dataset['Class_Bool'] = (
    # Returns true if class is Cammeo, and false if class is Osmancik
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)

# Split data
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

# Separate features and labels
label_columns = ['Class', 'Class_Bool']

train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a trained binary classification model for rice species.')
    parser.add_argument('--model_path', type=str, default='trained_model.keras', help='Path to load the trained model.')

    args = parser.parse_args()

    # Load the trained model
    loaded_model = keras.models.load_model(args.model_path)

    # Define input features (this should match the features used during training)
    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]

    # Evaluate the model on the validation set
    validation_metrics = loaded_model.evaluate(
        x={feature_name: np.array(validation_features[feature_name]) for feature_name in input_features},
        y=validation_labels,
        verbose=0
    )

    # Print the validation metrics
    print("Validation Metrics:")
    for metric_name, metric_value in zip(loaded_model.metrics_names, validation_metrics):
        print(f"{metric_name}: {metric_value:.4f}")