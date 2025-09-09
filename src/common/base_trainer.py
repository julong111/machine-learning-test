# -*- coding: utf-8 -*-
"""Defines the abstract base class for all model trainers."""

from abc import ABC, abstractmethod
import pandas as pd

from src.common.types import Experiment

class BaseTrainer(ABC):
    """Abstract base class for a model trainer.
    
    This class defines the standard interface for all trainer classes.
    Each specific trainer (e.g., for linear regression or logistic regression)
    should inherit from this class and implement its abstract methods.
    """

    @abstractmethod
    def build_model(self, **kwargs):
        """Builds the machine learning model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> Experiment:
        """Trains the model on the given data. Must be implemented by subclasses.

        Args:
            train_df: The DataFrame for training.
            val_df: The optional DataFrame for validation.

        Returns:
            An Experiment object containing the results of the training run.
        """
        pass

    @abstractmethod
    def evaluate(self, test_df: pd.DataFrame, **kwargs):
        """Evaluates the trained model on the test data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self, artifacts_dir: str, **kwargs):
        """Saves the trained model and any other artifacts. Must be implemented by subclasses."""
        pass
