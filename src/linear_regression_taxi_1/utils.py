# -*- coding: utf-8 -*-
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Internal utilities to replace the google-ml-edu dependency."""

import dataclasses
from typing import Any, List

import keras
import pandas as pd
from matplotlib import pyplot as plt


@dataclasses.dataclass
class ExperimentSettings:
  """A dataclass to hold experiment settings."""
  learning_rate: float
  number_epochs: int
  batch_size: int
  input_features: List[str]
  verbose: int = 1


@dataclasses.dataclass
class Experiment:
  """A dataclass to hold experiment results."""
  name: str
  settings: ExperimentSettings
  model: keras.Model
  epochs: list[int]
  metrics_history: pd.DataFrame


def plot_experiment_metrics(experiment: Experiment, metrics: List[str]):
  """Plots the training metrics for an experiment."""
  plt.figure(figsize=(15, 5))

  for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    plt.plot(
        experiment.epochs, experiment.metrics_history[metric], label='Training'
    )

    plt.legend()
    plt.title(f'{metric} vs. Epochs')

  plt.suptitle(f'Metrics for Experiment: {experiment.name}')


def predict_fare(
    model: keras.Model,
    dataset: pd.DataFrame,
    feature_names: List[str],
    label_name: str,
) -> pd.DataFrame:
    """Generates predictions for a dataset."""
    features = {name: dataset[name].values for name in feature_names}
    predictions = model.predict(features).flatten()

    prediction_df = pd.DataFrame()
    prediction_df["PREDICTED_FARE"] = predictions
    # Use .values to ignore the index from the sampled dataframe
    prediction_df["OBSERVED_FARE"] = dataset[label_name].values
    prediction_df["L1_LOSS"] = abs(
        prediction_df["PREDICTED_FARE"] - prediction_df["OBSERVED_FARE"]
    )
    for feature in feature_names:
        # Use .values here as well
        prediction_df[feature] = dataset[feature].values

    return prediction_df


def show_predictions(predictions: pd.DataFrame, title: str = "PREDICTIONS"):
    """Formats and prints a DataFrame of predictions.

    Args:
        predictions: DataFrame of predictions to display.
        title: The title to print for the table.
    """
    # Create a copy to avoid SettingWithCopyWarning when formatting
    display_df = predictions.copy()

    # Format currency columns
    for col in ["PREDICTED_FARE", "OBSERVED_FARE", "L1_LOSS"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")

    # Print the formatted table
    print("-" * 80)
    print("|" + title.center(78) + "|")
    print("-" * 80)
    print(display_df.to_string(index=False))
    print("\n")
