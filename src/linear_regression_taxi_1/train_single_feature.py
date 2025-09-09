# -*- coding: utf-8 -*-
"""Train a linear regression model with a single feature."""

from src.linear_regression_taxi_1 import utils, pipeline


def main():
    """Defines and runs the single-feature experiment."""
    # Define experiment settings
    experiment_name = "single_feature"
    settings = utils.ExperimentSettings(
        learning_rate=0.01,
        number_epochs=20,
        batch_size=1000,
        input_features=["TRIP_MILES"],
        verbose=1,
    )

    # Load and prepare data from the correct, new path
    training_df = pipeline.load_and_prepare_data(
        "artifacts/linear_regression_taxi_1/chicago_taxi_train.csv"
    )

    # Run the experiment
    pipeline.run_experiment(
        experiment_name=experiment_name,
        settings=settings,
        training_df=training_df,
        label_name="FARE",
    )


if __name__ == "__main__":
    main()
