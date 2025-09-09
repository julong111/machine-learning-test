# train.py

import argparse
import os
import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd

# Set the random seeds
keras.utils.set_random_seed(42)

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
artifacts_dir = os.path.join(project_root, 'artifacts', 'binary_classification_rice_2')
input_csv_path = os.path.join(artifacts_dir, 'Rice_Cammeo_Osmancik.csv')

# Load the dataset from the local path
rice_dataset_raw = pd.read_csv(input_csv_path)

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


# Define functions that build and train a model
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple classification model."""
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # Use a Concatenate layer to assemble the different inputs into a single
  # tensor which will be given as input to the Dense layer.
  # For example: [input_1[0][0], input_2[0][0]]

  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  # Call the compile method to transform the layers into a model that
  # Keras can execute.  Notice that we're using a different loss
  # function for classification than for regression.
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Feed a dataset into the model in order to train it."""

  # The x parameter of keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
      verbose=0 # Set verbose to 0 to avoid excessive output during training
  )

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a binary classification model for rice species.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')

    default_model_path = os.path.join(artifacts_dir, 'trained_model.keras')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to save the trained model.')

    args = parser.parse_args()

    # Define input features
    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]

    # Define experiment settings and metrics using command-line arguments
    settings = ml_edu.experiment.ExperimentSettings(
        learning_rate=args.learning_rate,
        number_epochs=args.epochs,
        batch_size=args.batch_size,
        classification_threshold=0.35, # Keep the default classification threshold
        input_features=input_features,
    )

    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy', threshold=settings.classification_threshold
        ),
        keras.metrics.Precision(
            name='precision', thresholds=settings.classification_threshold
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model = create_model(settings, metrics)

    # Train the model on the training set.
    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    # Ensure the output directory exists before saving
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Save the trained model
    model.save(args.model_path)

    print(f"Training complete and model saved to '{args.model_path}'")
