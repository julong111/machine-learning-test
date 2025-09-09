# -*- coding: utf-8 -*-
"""Common utilities for plotting model training history."""

import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np

# Import the Experiment dataclass from our new types module
from src.common.types import Experiment


def plot_training_history(experiment: Experiment, metrics: List[str], save_path: str):
    """Plots the training metrics for an experiment and saves the plot to a file."""
    print(f"\n--- Plotting training history for experiment: {experiment.name} ---")
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())

        if metric not in experiment.metrics_history.columns:
            print(f"Warning: Metric '{metric}' not found in experiment history. Skipping plot.")
            continue
            
        plt.plot(experiment.epochs, experiment.metrics_history[metric], label='Training')

        val_metric = f"val_{metric}"
        if val_metric in experiment.metrics_history.columns:
            plt.plot(experiment.epochs, experiment.metrics_history[val_metric], label='Validation')

        plt.legend()
        plt.title(f'{metric.capitalize()} vs. Epochs')

    plt.suptitle(f'Metrics for Experiment: {experiment.name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path)
    plt.clf() # Clear the figure to free memory
    print("Plot saved successfully.")

def plot_classification_history(experiment: Experiment, save_path: str):
    """Plots a 2x2 grid of common classification metrics and saves the plot."""
    print(f"\n--- Plotting classification history for experiment: {experiment.name} ---")
    history_df = experiment.metrics_history

    required_metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for metric in required_metrics:
        if metric not in history_df.columns:
            print(f"Warning: Metric '{metric}' not found. Cannot generate classification plot.")
            return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Training History: {experiment.name}', fontsize=16)

    # Plot Loss
    axes[0, 0].plot(history_df['loss'], label='Training Loss')
    if 'val_loss' in history_df.columns:
        axes[0, 0].plot(history_df['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss vs. Epochs')
    axes[0, 0].legend()

    # Plot Accuracy
    axes[0, 1].plot(history_df['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_df.columns:
        axes[0, 1].plot(history_df['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy vs. Epochs')
    axes[0, 1].legend()

    # Plot Precision & Recall
    axes[1, 0].plot(history_df['precision'], label='Training Precision')
    if 'val_precision' in history_df.columns:
        axes[1, 0].plot(history_df['val_precision'], label='Validation Precision')
    axes[1, 0].plot(history_df['recall'], label='Training Recall', linestyle='--')
    if 'val_recall' in history_df.columns:
        axes[1, 0].plot(history_df['val_recall'], label='Validation Recall', linestyle='--')
    axes[1, 0].set_title('Precision & Recall vs. Epochs')
    axes[1, 0].legend()

    # Plot AUC
    axes[1, 1].plot(history_df['auc'], label='Training AUC')
    if 'val_auc' in history_df.columns:
        axes[1, 1].plot(history_df['val_auc'], label='Validation AUC')
    axes[1, 1].set_title('AUC vs. Epochs')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving classification plot to {save_path}...")
    plt.savefig(save_path)
    plt.clf()
    print("Plot saved successfully.")


def plot_precision_recall_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    optimal_idx: int,
    current_threshold: float,
    save_path: str
):
    """
    Plots the precision-recall curve and highlights the optimal and current thresholds.

    Args:
        precisions: An array of precision values.
        recalls: An array of recall values.
        thresholds: An array of threshold values corresponding to precision and recall.
        optimal_idx: The index of the optimal threshold in the arrays.
        current_threshold: The threshold currently used by the model for decisions.
        save_path: The file path to save the plot.
    """
    print(f"\n--- Plotting Precision-Recall Curve ---")
    plt.figure(figsize=(10, 8))

    # Plot the main PR curve
    plt.plot(recalls, precisions, label='Precision-Recall Curve')

    # Highlight the optimal threshold point
    optimal_recall = recalls[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_thresh_val = thresholds[optimal_idx]
    plt.plot(optimal_recall, optimal_precision, 'ro', label=f'Optimal (F2-Score) @ Threshold={optimal_thresh_val:.2f}')
    plt.annotate(
        f'P={optimal_precision:.2f}, R={optimal_recall:.2f}',
        xy=(optimal_recall, optimal_precision),
        xytext=(optimal_recall + 0.02, optimal_precision - 0.04)
    )

    # Highlight the current threshold point
    # Find the closest threshold in the list to the current one
    current_idx = np.argmin(np.abs(thresholds - current_threshold))
    current_recall = recalls[current_idx]
    current_precision = precisions[current_idx]
    plt.plot(current_recall, current_precision, 'go', label=f'Current @ Threshold={current_threshold:.2f}')
    plt.annotate(
        f'P={current_precision:.2f}, R={current_recall:.2f}',
        xy=(current_recall, current_precision),
        xytext=(current_recall + 0.02, current_precision - 0.04)
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Analysis')
    plt.legend()
    plt.grid(True)

    print(f"Saving PR curve plot to '{save_path}'")
    plt.savefig(save_path)
    plt.clf()
    print("Plot saved successfully.")
