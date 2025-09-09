# -*- coding: utf-8 -*-
"""Common utilities for plotting model training history."""

import os
from typing import List
from matplotlib import pyplot as plt

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
