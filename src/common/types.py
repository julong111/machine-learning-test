# -*- coding: utf-8 -*-
"""Common type definitions and data classes for the project."""

import dataclasses
from typing import List, Dict

import keras
import pandas as pd


@dataclasses.dataclass
class ExperimentSettings:
  """A dataclass to hold experiment settings."""
  learning_rate: float
  number_epochs: int
  batch_size: int
  input_features: List[str]
  decision_threshold: float = 0.5
  auc_num_thresholds: int = 100
  verbose: int = 1


@dataclasses.dataclass
class Experiment:
  """A dataclass to hold experiment results."""
  name: str
  settings: ExperimentSettings
  model: keras.Model
  epochs: list[int]
  metrics_history: pd.DataFrame
