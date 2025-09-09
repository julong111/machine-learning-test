
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

import keras
import pandas as pd
from matplotlib import pyplot as plt

from src.linear_regression_taxi_1 import utils

class BaseTrainer(ABC):
    """
    Abstract base class for a trainer.
    """
    def __init__(self, *args, **kwargs):
        # Allow subclasses to have their own constructors
        pass

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model.
        """
        pass

class LinearRegressionTrainer(BaseTrainer):
    """
    Trainer for the linear regression model.
    """
    def __init__(self, experiment_name: str, settings: utils.ExperimentSettings, training_df: pd.DataFrame, label_name: str):
        super().__init__()
        self.experiment_name = experiment_name
        self.settings = settings
        self.training_df = training_df
        self.label_name = label_name
        self.model: keras.Model = None
        self.experiment: utils.Experiment = None

    def _create_model(self, metrics: list[keras.metrics.Metric]) -> keras.Model:
        """Create and compile a simple linear regression model."""
        inputs = {
            name: keras.Input(shape=(1,), name=name)
            for name in self.settings.input_features
        }
        concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
        outputs = keras.layers.Dense(units=1)(concatenated_inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=self.settings.learning_rate),
            loss="mean_squared_error",
            metrics=metrics,
        )
        return model

    def _train_model(self, model: keras.Model) -> utils.Experiment:
        """Train the model by feeding it data."""
        features = {name: self.training_df[name].values for name in self.settings.input_features}
        label = self.training_df[self.label_name].values
        history = model.fit(
            x=features,
            y=label,
            batch_size=self.settings.batch_size,
            epochs=self.settings.number_epochs,
            verbose=self.settings.verbose,
        )

        return utils.Experiment(
            name=self.experiment_name,
            settings=self.settings,
            model=model,
            epochs=history.epoch,
            metrics_history=pd.DataFrame(history.history),
        )

    def train(self):
        """Runs a single experiment and saves the resulting artifacts."""
        print(f"--- Running experiment: {self.experiment_name} ---")

        metrics = [keras.metrics.RootMeanSquaredError(name="rmse")]
        self.model = self._create_model(metrics)
        self.experiment = self._train_model(self.model)

        # Point to the new, correct artifacts subdirectory
        artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / f"{self.experiment_name}_model.keras"
        settings_path = artifacts_dir / f"{self.experiment_name}_settings.pkl"

        self.model.save(model_path)
        print(f"Trained model saved to '{model_path}'")

        with open(settings_path, "wb") as f:
            pickle.dump(self.settings, f)
        print(f"Model settings saved to '{settings_path}'")

    def evaluate(self):
        """Plots the experiment metrics."""
        if not self.experiment:
            print("Model not trained yet. Call train() first.")
            return

        artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        plot_path = artifacts_dir / f"{self.experiment_name}_rmse_plot.png"

        utils.plot_experiment_metrics(self.experiment, ["rmse"])
        plt.savefig(plot_path)
        plt.clf()
        print(f"Training plot saved to '{plot_path}'\n")
