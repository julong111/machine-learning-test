# -*- coding: utf-8 -*-
"""
A dedicated script for hyperparameter tuning of the taxi fare prediction models
using Keras Tuner.
"""

import os
import json
import keras
from keras.callbacks import EarlyStopping
import keras_tuner as kt

# Add the project root to the Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in os.sys.path:
    os.sys.path.insert(0, project_root)

# Import project-specific utilities
from src.common import csv_utils
from src.common import keras_utils
from src.linear_regression_taxi_1.taxi_training import MODEL_CONFIGS


class TaxiHyperModel(kt.HyperModel):
    """
    A HyperModel class for Keras Tuner, which defines how to build the model
    and how to fit it, allowing tuning of both model and training hyperparameters.
    """
    def __init__(self, input_features):
        self.input_features = input_features

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Builds the model and defines the hyperparameter search space for it.

        Args:
            hp: A HyperParameters object from Keras Tuner.

        Returns:
            A compiled Keras model.
        """
        # Define the search space for the learning rate.
        hp_learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-4,
            max_value=1e-1,
            sampling='log'
        )

        # Reuse our existing model creation utility
        model = keras_utils.create_linear_regression_model(
            input_features=self.input_features,
            learning_rate=hp_learning_rate
        )
        return model

    def fit(self, hp: kt.HyperParameters, model: keras.Model, *args, **kwargs):
        """
        Overrides the default fit method to tune training-related hyperparameters
        like batch_size and to inject callbacks like EarlyStopping.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,  # Stop if val_loss doesn't improve for 5 epochs
            restore_best_weights=True # Restore model weights from the epoch with the best val_loss
        )

        # This is the key change to fix the TypeError. We safely remove
        # 'callbacks' from kwargs, add our EarlyStopping, and then pass the
        # consolidated list explicitly.
        callbacks = kwargs.pop('callbacks', []) # Use .pop() instead of .get()
        callbacks.append(early_stopping)

        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[32, 64, 128]),
            callbacks=callbacks,
            **kwargs
        )


def main():
    """Orchestrates the hyperparameter tuning process."""
    print("\n--- Starting Comprehensive Hyperparameter Tuning for smart_model ---")

    # 1. Load the pre-processed and scaled data
    artifacts_dir = os.path.join(project_root, 'artifacts', 'linear_regression_taxi_1')
    train_df = csv_utils.load_csv(os.path.join(artifacts_dir, "chicago_taxi_train.csv"))
    val_df = csv_utils.load_csv(os.path.join(artifacts_dir, "chicago_taxi_validation.csv"))

    smart_model_features = MODEL_CONFIGS['smart_model'].input_features

    # Prepare data for Keras
    train_features, train_labels = keras_utils.prepare_features_and_labels(
        train_df, smart_model_features, 'FARE'
    )
    val_features, val_labels = keras_utils.prepare_features_and_labels(
        val_df, smart_model_features, 'FARE'
    )

    # 2. Initialize and run the tuner
    hypermodel = TaxiHyperModel(input_features=smart_model_features)
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory=os.path.join(artifacts_dir, 'tuning'),
        project_name='taxi_fare_comprehensive_tuning',
        overwrite=True # Overwrite previous tuning results
    )

    # The number of epochs here is the max possible; EarlyStopping will handle the rest.
    tuner.search(train_features, train_labels, epochs=20, validation_data=(val_features, val_labels), verbose=2)

    # 3. Print the results
    print("\n--- Tuning Complete ---")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_learning_rate = best_hps.get('learning_rate')
    best_batch_size = best_hps.get('batch_size')

    print(f"Optimal learning rate found: {best_learning_rate:.6f}")
    print(f"Optimal batch size found:    {best_batch_size}")

    # 4. Save the best hyperparameters to a file
    best_params = {
        'learning_rate': best_learning_rate,
        'batch_size': best_batch_size
    }
    config_path = os.path.join(artifacts_dir, 'best_hyperparameters.json')
    with open(config_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\nBest hyperparameters saved to {config_path}")
    print("The training script will now automatically use these values.")


if __name__ == "__main__":
    main()
