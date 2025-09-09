# -*- coding: utf-8 -*-
"""Concrete trainer for linear regression models."""

import os
import pandas as pd

# Import from our common modules
from src.common.base_trainer import BaseTrainer
from src.common.types import ExperimentSettings
from src.common import keras_utils
from src.common import plot_utils
from src.common import prediction_utils


class LinearRegressionTrainer(BaseTrainer):
    """
    A concrete trainer for building, training, evaluating, and saving
    linear regression models by inheriting from the BaseTrainer.
    """

    def __init__(self, experiment_name: str, settings: ExperimentSettings):
        """Initializes the linear regression trainer."""
        # The label column is fixed for this dataset
        super().__init__(experiment_name, settings, label_column="FARE")

    def build_model(self):
        """Builds the Keras linear regression model."""
        self.model = keras_utils.create_linear_regression_model(
            input_features=self.settings.input_features,
            learning_rate=self.settings.learning_rate
        )

    def evaluate(self, test_df: pd.DataFrame, large_error_threshold: float = 5.0):
        """Evaluates the model on the test set and prints a report."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Evaluating model on the test set ---")
        
        all_predictions_df = prediction_utils.predict_on_dataframe(
            model=self.model,
            dataset=test_df,
            feature_names=self.settings.input_features,
            label_name=self.label_column,
            prediction_col_name="PREDICTED_FARE",
            observed_col_name="OBSERVED_FARE",
            error_col_name="L1_LOSS"
        )

        features = {name: test_df[name].values for name in self.settings.input_features}
        label = test_df[self.label_column].values
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
        
        sorted_predictions = all_predictions_df.sort_values(by="L1_LOSS", ascending=False)
        
        worst_predictions = sorted_predictions.head(10)
        prediction_utils.show_predictions(
            worst_predictions, 
            title="TOP 10 WORST PREDICTIONS (HIGHEST ERROR)",
            format_cols={"PREDICTED_FARE": "${:.2f}", "OBSERVED_FARE": "${:.2f}", "L1_LOSS": "${:.2f}"}
        )

        best_predictions = sorted_predictions.tail(5).sort_values(by="L1_LOSS", ascending=True)
        prediction_utils.show_predictions(
            best_predictions, 
            title="TOP 10 BEST PREDICTIONS (LOWEST ERROR)",
            format_cols={"PREDICTED_FARE": "${:.2f}", "OBSERVED_FARE": "${:.2f}", "L1_LOSS": "${:.2f}"}
        )

    def _plot_history(self, artifacts_dir: str):
        """Saves a plot of the model's training history (RMSE)."""
        if self.experiment is None:
            raise RuntimeError("Cannot plot history before training.")
            
        plot_path = os.path.join(artifacts_dir, f"{self.experiment_name}_rmse_plot.png")
        print(f"Saving training history plot to '{plot_path}'")
        plot_utils.plot_training_history(self.experiment, ["rmse"], plot_path)
