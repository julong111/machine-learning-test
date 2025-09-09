# -*- coding: utf-8 -*-
"""Shared pipeline functions for model training."""

import pickle
from pathlib import Path

import keras
import pandas as pd
from matplotlib import pyplot as plt

from src.linear_regression_taxi_1 import utils


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Loads and preprocesses the taxi fare dataset."""
    try:
        dataset = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{Path(csv_path).name}' not found.")
        print("Please run the script from the project root directory.")
        exit()

    # Keep only the necessary columns and create the TRIP_MINUTES feature
    df = dataset.loc[:, ("TRIP_MILES", "TRIP_SECONDS", "FARE")]
    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, "TRIP_MINUTES"] = df["TRIP_SECONDS"] / 60

    print("Dataset loaded and prepared successfully.")
    print(f"Total number of rows: {len(df.index)}\n")
    return df


def create_model(
    settings: utils.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple linear regression model."""
    inputs = {
        name: keras.Input(shape=(1,), name=name)
        for name in settings.input_features
    }
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
        loss="mean_squared_error",
        metrics=metrics,
    )
    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: utils.ExperimentSettings,
) -> utils.Experiment:
    """Train the model by feeding it data."""
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(
        x=features,
        y=label,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        verbose=settings.verbose,
    )

    return utils.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


def run_experiment(
    experiment_name: str,
    settings: utils.ExperimentSettings,
    training_df: pd.DataFrame,
    label_name: str,
):
    """Runs a single experiment and saves the resulting artifacts."""
    print(f"--- Running experiment: {experiment_name} ---")

    metrics = [keras.metrics.RootMeanSquaredError(name="rmse")]
    model = create_model(settings, metrics)
    experiment = train_model(
        experiment_name, model, training_df, label_name, settings
    )

    # Point to the new, correct artifacts subdirectory
    artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"{experiment_name}_model.keras"
    settings_path = artifacts_dir / f"{experiment_name}_settings.pkl"
    plot_path = artifacts_dir / f"{experiment_name}_rmse_plot.png"

    model.save(model_path)
    print(f"Trained model saved to '{model_path}'")

    with open(settings_path, "wb") as f:
        pickle.dump(settings, f)
    print(f"Model settings saved to '{settings_path}'")

    utils.plot_experiment_metrics(experiment, ["rmse"])
    plt.savefig(plot_path)
    plt.clf()
    print(f"Training plot saved to '{plot_path}'\n")
