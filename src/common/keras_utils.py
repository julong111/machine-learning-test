# -*- coding: utf-8 -*-
"""Common utilities for creating Keras models."""

from typing import List, Dict
import keras
import numpy as np
import pandas as pd

def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Prepares features and labels for a Keras model from a DataFrame."""
    if not all(col in df.columns for col in feature_cols):
        raise ValueError("One or more feature columns are not in the DataFrame.")
    if label_col not in df.columns:
        raise ValueError("The label column is not in the DataFrame.")
        
    labels = df[label_col].to_numpy()
    features = {name: np.array(df[name]) for name in feature_cols}
    return features, labels

def create_linear_regression_model(
    input_features: List[str],
    learning_rate: float
) -> keras.Model:
    """Create and compile a simple Keras linear regression model."""
    print("\n--- Creating Keras linear regression model ---")
    
    inputs = {name: keras.Input(shape=(1,), name=name) for name in input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    
    print("Model created and compiled successfully.")
    return model

def create_binary_classification_model(
    input_features: List[str],
    learning_rate: float,
    metrics_thresholds: Dict[str, float],
    auc_num_thresholds: int,
) -> keras.Model:
    """Create and compile a simple Keras binary classification model."""
    print("\n--- Creating Keras binary classification model ---")
    
    inputs = {name: keras.Input(shape=(1,), name=name) for name in input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1, activation=keras.activations.sigmoid)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=metrics_thresholds.get('accuracy', 0.5)),
            keras.metrics.Precision(name='precision', thresholds=metrics_thresholds.get('precision', 0.5)),
            keras.metrics.Recall(name='recall', thresholds=metrics_thresholds.get('recall', 0.5)),
            keras.metrics.AUC(num_thresholds=auc_num_thresholds, name='auc'),
        ],
    )
    
    print("Model created and compiled successfully.")
    return model
