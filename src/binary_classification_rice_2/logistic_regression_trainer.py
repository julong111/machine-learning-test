# -*- coding: utf-8 -*-
"""Concrete trainer for logistic regression models."""

import os
import pickle
import pandas as pd
import keras

# Import from our common modules
from src.common.base_trainer import BaseTrainer
from src.common.types import Experiment, ExperimentSettings
from src.common import keras_utils
from src.common import plot_utils
from src.common import prediction_utils

class LogisticRegressionTrainer(BaseTrainer):
    """
    A concrete trainer for building, training, evaluating, and saving
    logistic regression models using Keras.
    """

    def __init__(self, experiment_name: str, settings: ExperimentSettings):
        """Initializes the logistic regression trainer."""
        self.experiment_name = experiment_name
        self.settings = settings
        self.model: keras.Model | None = None
        self.experiment: Experiment | None = None

    def build_model(self):
        """Builds the Keras binary classification model."""
        self.model = keras_utils.create_binary_classification_model(
            input_features=self.settings.input_features,
            learning_rate=self.settings.learning_rate
        )

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> Experiment:
        """Trains the model and returns the experiment results."""
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        print(f"\n--- Training model: {self.experiment_name} ---")
        features = {name: train_df[name].values for name in self.settings.input_features}
        label = train_df["Class_Bool"].values

        validation_data = None
        if val_df is not None:
            val_features = {name: val_df[name].values for name in self.settings.input_features}
            val_label = val_df["Class_Bool"].values
            validation_data = (val_features, val_label)

        history = self.model.fit(
            x=features,
            y=label,
            batch_size=self.settings.batch_size,
            epochs=self.settings.number_epochs,
            verbose=self.settings.verbose,
            validation_data=validation_data
        )

        self.experiment = Experiment(
            name=self.experiment_name,
            settings=self.settings,
            model=self.model,
            epochs=history.epoch,
            metrics_history=pd.DataFrame(history.history),
        )
        print("--- Model training complete. ---")
        return self.experiment

    def evaluate(self, test_df: pd.DataFrame):
        """Evaluates the model on the test set and prints a report."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Evaluating model on the test set ---")
        
        features = {name: test_df[name].values for name in self.settings.input_features}
        label = test_df["Class_Bool"].values
        results = self.model.evaluate(features, label, verbose=0)

        print(f"\n--- Overall Test Results for: {self.experiment_name} ---")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"{metric_name.capitalize()}: {value:.4f}")
        print("-" * 35 + "\n")

        # Optionally show predictions for classification as well
        all_predictions_df = prediction_utils.predict_on_dataframe(
            model=self.model,
            dataset=test_df,
            feature_names=self.settings.input_features,
            label_name="Class_Bool",
            prediction_col_name="PREDICTED_PROBABILITY",
            observed_col_name="OBSERVED_CLASS",
            error_col_name="PREDICTION_ERROR" # For binary classification, this might be less intuitive
        )
        # For binary classification, we might want to show predicted class (0/1) instead of probability
        all_predictions_df["PREDICTED_CLASS"] = (all_predictions_df["PREDICTED_PROBABILITY"] > 0.5).astype(int)
        
        prediction_utils.show_predictions(
            all_predictions_df.head(10), 
            title="Sample Predictions (First 10)",
            format_cols={
                "PREDICTED_PROBABILITY": "{:.4f}", 
                "OBSERVED_CLASS": "{}", 
                "PREDICTED_CLASS": "{}"
            }
        )


    def save(self, artifacts_dir: str):
        """Saves the trained model, settings, and training plot."""
        if self.model is None or self.experiment is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Saving artifacts for experiment: {self.experiment_name} ---")
        
        model_path = os.path.join(artifacts_dir, f"{self.experiment_name}_model.keras")
        settings_path = os.path.join(artifacts_dir, f"{self.experiment_name}_settings.pkl")
        plot_path = os.path.join(artifacts_dir, f"{self.experiment_name}_training_history.png")

        self.model.save(model_path)
        print(f"Trained model saved to '{model_path}'")

        with open(settings_path, "wb") as f:
            pickle.dump(self.settings, f)
        print(f"Model settings saved to '{settings_path}'")

        plot_utils.plot_classification_history(self.experiment, plot_path)
