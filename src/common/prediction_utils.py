# -*- coding: utf-8 -*-
"""Common utilities for making predictions and displaying results."""

from typing import List, Dict
import keras
import pandas as pd

def predict_on_dataframe(
    model: keras.Model,
    dataset: pd.DataFrame,
    feature_names: List[str],
    label_name: str,
    prediction_col_name: str = "prediction",
    observed_col_name: str = "observed",
    error_col_name: str = "error"
) -> pd.DataFrame:
    """Generates predictions for a dataset.

    Args:
        model: The trained Keras model.
        dataset: The DataFrame to make predictions on.
        feature_names: A list of column names to be used as input features.
        label_name: The name of the true label column in the dataset.
        prediction_col_name: The name for the new column for predictions.
        observed_col_name: The name for the new column for the true label.
        error_col_name: The name for the new column for the absolute error.

    Returns:
        A new DataFrame with features, observed values, predictions, and error.
    """
    print(f"\n--- Generating predictions for {len(dataset)} records ---")
    features = {name: dataset[name].values for name in feature_names}
    predictions = model.predict(features).flatten()

    prediction_df = pd.DataFrame()
    prediction_df[prediction_col_name] = predictions
    # Use .values to ignore the index from a potentially sampled dataframe
    prediction_df[observed_col_name] = dataset[label_name].values
    prediction_df[error_col_name] = abs(
        prediction_df[prediction_col_name] - prediction_df[observed_col_name]
    )
    # Also include the input features for context
    for feature in feature_names:
        prediction_df[feature] = dataset[feature].values

    print("Predictions generated successfully.")
    return prediction_df


def show_predictions(
    predictions_df: pd.DataFrame, 
    title: str = "PREDICTIONS",
    format_cols: Dict[str, str] | None = None
):
    """Formats and prints a DataFrame of predictions.

    Args:
        predictions_df: DataFrame of predictions to display.
        title: The title to print for the table.
        format_cols: A dictionary mapping column names to format strings, 
                     e.g., {'price': '${:.2f}'}.
    """
    # Create a copy to avoid SettingWithCopyWarning when formatting
    display_df = predictions_df.copy()

    # Apply formatting if specified
    if format_cols:
        for col, format_spec in format_cols.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format_spec.format(x))

    # Print the formatted table
    print("-" * 80)
    print("|" + title.center(78) + "|")
    print("-" * 80)
    print(display_df.to_string(index=False))
    print("\n")
