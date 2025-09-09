# -*- coding: utf-8 -*-
"""Concrete trainer for logistic regression models."""

import os
import numpy as np
import pandas as pd

# Import from our common modules
from src.common.base_trainer import BaseTrainer
from src.common.types import ExperimentSettings
from src.common import keras_utils
from src.common import plot_utils
from src.common import prediction_utils

# Import for advanced evaluation
from sklearn.metrics import precision_recall_curve


class LogisticRegressionTrainer(BaseTrainer):
    """
    A concrete trainer for building, training, evaluating, and saving
    logistic regression models by inheriting from the BaseTrainer.
    """

    def __init__(self, experiment_name: str, settings: ExperimentSettings):
        """Initializes the logistic regression trainer."""
        # The label column is fixed for this dataset
        super().__init__(experiment_name, settings, label_column="Class_Bool")

    def build_model(self):
        """Builds the Keras binary classification model."""
        self.model = keras_utils.create_binary_classification_model(
            input_features=self.settings.input_features,
            learning_rate=self.settings.learning_rate,
            decision_threshold=self.settings.decision_threshold,
            auc_num_thresholds=self.settings.auc_num_thresholds,
        )

    def evaluate(self, test_df: pd.DataFrame, artifacts_dir: str):
        """Evaluates the model on the test set and prints a report."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Evaluating model on the test set ---")
        
        features = {name: test_df[name].values for name in self.settings.input_features}
        label = test_df[self.label_column].values
        results = self.model.evaluate(features, label, verbose=0)

        print(f"\n--- Overall Test Results for: {self.experiment_name} ---")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"{metric_name.capitalize()}: {value:.4f}")
        print("-" * 35 + "\n")

        all_predictions_df = prediction_utils.predict_on_dataframe(
            model=self.model,
            dataset=test_df,
            feature_names=self.settings.input_features,
            label_name=self.label_column,
            prediction_col_name="PREDICTED_PROBABILITY",
            observed_col_name="OBSERVED_CLASS",
            error_col_name="PREDICTION_ERROR"
        )
        # Use the primary decision threshold for final classification.
        threshold = self.settings.decision_threshold
        all_predictions_df["PREDICTED_CLASS"] = (all_predictions_df["PREDICTED_PROBABILITY"] > threshold).astype(int)
        
        self._analyze_and_plot_pr_curve(all_predictions_df, artifacts_dir)

        prediction_utils.show_predictions(
            all_predictions_df.head(10), 
            title="Sample Predictions (First 10)",
            format_cols={
                "PREDICTED_PROBABILITY": "{:.4f}", 
                "OBSERVED_CLASS": "{}", 
                "PREDICTED_CLASS": "{}"
            }
        )

    def _plot_history(self, artifacts_dir: str):
        """Saves a plot of the model's training history (classification metrics)."""
        if self.experiment is None:
            raise RuntimeError("Cannot plot history before training.")
            
        plot_path = os.path.join(artifacts_dir, f"{self.experiment_name}_training_history.png")
        print(f"Saving training history plot to '{plot_path}'")
        plot_utils.plot_classification_history(self.experiment, plot_path)

    def _analyze_and_plot_pr_curve(self, predictions_df: pd.DataFrame, artifacts_dir: str):
        """
        Analyzes the precision-recall curve to find the optimal threshold and plots it.

        Args:
            predictions_df: DataFrame with 'OBSERVED_CLASS' and 'PREDICTED_PROBABILITY'.
            artifacts_dir: Directory to save the plot.
        """
        y_true = predictions_df["OBSERVED_CLASS"]
        y_pred_probs = predictions_df["PREDICTED_PROBABILITY"]

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)

        # Calculate F2-score for each threshold. F2 gives more weight to recall.
        # We slice precisions and recalls to match the length of thresholds.
        f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1])
        # Handle the case where the denominator is zero
        f2_scores = np.nan_to_num(f2_scores)

        # Find the threshold that maximizes the F2-score
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_precision = precisions[optimal_idx]
        optimal_recall = recalls[optimal_idx]

        print("\n--- Precision-Recall Curve Analysis ---")
        print(f"Optimal threshold (for max F2-score): {optimal_threshold:.4f}")
        print(f"  - Precision at this threshold: {optimal_precision:.4f}")
        print(f"  - Recall at this threshold:    {optimal_recall:.4f}")
        print("-" * 35 + "\n")

        # Plot the curve
        plot_path = os.path.join(artifacts_dir, f"{self.experiment_name}_pr_curve.png")
        plot_utils.plot_precision_recall_curve(
            precisions=precisions,
            recalls=recalls,
            thresholds=thresholds,
            optimal_idx=optimal_idx,
            current_threshold=self.settings.decision_threshold,
            save_path=plot_path
        )
