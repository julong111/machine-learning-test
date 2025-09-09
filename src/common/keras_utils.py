# -*- coding: utf-8 -*-
"""Common utilities for creating Keras models."""

from typing import List
import keras

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
    learning_rate: float
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
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            keras.metrics.Precision(name='precision', thresholds=0.5),
            keras.metrics.Recall(name='recall', thresholds=0.5),
            keras.metrics.AUC(num_thresholds=100, name='auc'),
        ],
    )
    
    print("Model created and compiled successfully.")
    return model
