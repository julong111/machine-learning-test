# -*- coding: utf-8 -*-
"""Concrete trainer for linear regression models."""

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

class LinearRegressionTrainer(BaseTrainer):
    """
    A concrete trainer for building, training, evaluating, and saving
    linear regression models using Keras.
    """

    def __init__(self, experiment_name: str, settings: ExperimentSettings):
        """Initializes the linear regression trainer."""
        self.experiment_name = experiment_name
        self.settings = settings
        self.model: keras.Model | None = None
        self.experiment: Experiment | None = None

    def build_model(self):
        """Builds the Keras linear regression model."""
        self.model = keras_utils.create_linear_regression_model(
            input_features=self.settings.input_features,
            learning_rate=self.settings.learning_rate
        )

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> Experiment:
        """Trains the model and returns the experiment results."""
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        print(f"\n--- Training model: {self.experiment_name} ---")
        features = {name: train_df[name].values for name in self.settings.input_features}
        label = train_df["FARE"].values

        validation_data = None
        if val_df is not None:
            val_features = {name: val_df[name].values for name in self.settings.input_features}
            val_label = val_df["FARE"].values
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

    def evaluate(self, test_df: pd.DataFrame, large_error_threshold: float = 5.0):
        """Evaluates the model on the test set and prints a report."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Evaluating model on the test set ---")
        
        all_predictions_df = prediction_utils.predict_on_dataframe(
            model=self.model,
            dataset=test_df,
            feature_names=self.settings.input_features,
            label_name="FARE",
            prediction_col_name="PREDICTED_FARE",
            observed_col_name="OBSERVED_FARE",
            error_col_name="L1_LOSS"
        )

        features = {name: test_df[name].values for name in self.settings.input_features}
        label = test_df["FARE"].values
        results = self.model.evaluate(features, label, verbose=0)

        large_errors_df = all_predictions_df[all_predictions_df['L1_LOSS'] > large_error_threshold]
        large_error_count = len(large_errors_df)
        total_predictions = len(all_predictions_df)
        large_error_percentage = (large_error_count / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"\n--- Overall Test Results for: {self.experiment_name} ---")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"{metric_name.capitalize()}: {value:.4f}")
        print("-" * 35)
        print(f"Errors > ${large_error_threshold:.2f}: {large_error_count} / {total_predictions} ({large_error_percentage:.2f}%)")
        print("-" * 35 + "\n")
        
        # Sort by error to show worst and best predictions
        sorted_predictions = all_predictions_df.sort_values(by="L1_LOSS", ascending=False)
        
        # Show the 10 worst predictions
        worst_predictions = sorted_predictions.head(10)
        prediction_utils.show_predictions(
            worst_predictions, 
            title="TOP 10 WORST PREDICTIONS (HIGHEST ERROR)",
            format_cols={"PREDICTED_FARE": "${:.2f}", "OBSERVED_FARE": "${:.2f}", "L1_LOSS": "${:.2f}"}
        )

        # Show the 5 best predictions
        best_predictions = sorted_predictions.tail(5).sort_values(by="L1_LOSS", ascending=True)
        prediction_utils.show_predictions(
            best_predictions, 
            title="TOP 10 BEST PREDICTIONS (LOWEST ERROR)",
            format_cols={"PREDICTED_FARE": "${:.2f}", "OBSERVED_FARE": "${:.2f}", "L1_LOSS": "${:.2f}"}
        )


    def save(self, artifacts_dir: str):
        """Saves the model, settings, and training plot."""
        if self.model is None or self.experiment is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Saving artifacts for experiment: {self.experiment_name} ---")
        
        model_path = os.path.join(artifacts_dir, f"{self.experiment_name}_model.keras")
        settings_path = os.path.join(artifacts_dir, f"{self.experiment_name}_settings.pkl")
        plot_path = os.path.join(artifacts_dir, f"{self.experiment_name}_rmse_plot.png")

        self.model.save(model_path)
        print(f"Trained model saved to '{model_path}'")

        with open(settings_path, "wb") as f:
            pickle.dump(self.settings, f)
        print(f"Model settings saved to '{settings_path}'")

        plot_utils.plot_training_history(self.experiment, ["rmse"], plot_path)
