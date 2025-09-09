# -*- coding: utf-8 -*-
"""Defines the abstract base class for all model trainers."""

import os
import pickle
import pandas as pd
import keras
from abc import ABC, abstractmethod

from src.common.types import Experiment, ExperimentSettings

class BaseTrainer(ABC):
    """
    Abstract base class for a model trainer.

    This class provides a standard structure for training, evaluating, and
    saving a model. It handles the common logic for training loops and saving
    artifacts, while leaving model-specific implementations to subclasses.
    """

    def __init__(self, experiment_name: str, settings: ExperimentSettings, label_column: str):
        """Initializes the base trainer."""
        self.experiment_name = experiment_name
        self.settings = settings
        self.label_column = label_column
        self.model: keras.Model | None = None
        self.experiment: Experiment | None = None

    @abstractmethod
    def build_model(self):
        """Builds the Keras model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate(self, test_df: pd.DataFrame, **kwargs):
        """Evaluates the trained model on the test data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _plot_history(self, artifacts_dir: str):
        """Saves a plot of the model's training history. Must be implemented by subclasses."""
        pass

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> Experiment:
        """Trains the model and returns the experiment results."""
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        print(f"\n--- Training model: {self.experiment_name} ---")
        features = {name: train_df[name].values for name in self.settings.input_features}
        label = train_df[self.label_column].values

        validation_data = None
        if val_df is not None:
            val_features = {name: val_df[name].values for name in self.settings.input_features}
            val_label = val_df[self.label_column].values
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

    def save(self, artifacts_dir: str):
        """Saves the model, settings, and training plot."""
        if self.model is None or self.experiment is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        print(f"\n--- Saving artifacts for experiment: {self.experiment_name} ---")
        
        model_path = os.path.join(artifacts_dir, f"{self.experiment_name}_model.keras")
        settings_path = os.path.join(artifacts_dir, f"{self.experiment_name}_settings.pkl")

        self.model.save(model_path)
        print(f"Trained model saved to '{model_path}'")

        with open(settings_path, "wb") as f:
            pickle.dump(self.settings, f)
        print(f"Model settings saved to '{settings_path}'")

        # Delegate plotting to the subclass
        self._plot_history(artifacts_dir)
