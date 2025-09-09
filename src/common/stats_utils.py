# -*- coding: utf-8 -*-
"""Common utilities for statistical analysis, such as outlier detection."""

import pandas as pd

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the numerical features of a DataFrame using z-score normalization.

    Args:
        df: The pandas DataFrame to normalize.

    Returns:
        A new DataFrame with the numerical features normalized.
    """
    print("\n--- Normalizing numerical features ---")
    numerical_features = df.select_dtypes(include='number').columns
    
    if not numerical_features.any():
        print("No numerical features to normalize.")
        return df.copy()

    feature_mean = df[numerical_features].mean()
    feature_std = df[numerical_features].std()

    # Avoid division by zero for columns with no variance
    feature_std[feature_std == 0] = 1.0

    normalized_df = df.copy()
    normalized_df[numerical_features] = (df[numerical_features] - feature_mean) / feature_std

    print("Numerical features normalized successfully.")
    return normalized_df

def get_iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Calculates the lower and upper bounds for outlier detection using the IQR method.

    Args:
        series: The pandas Series to analyze.
        multiplier: The multiplier for the IQR. A standard value is 1.5.

    Returns:
        A tuple containing the lower and upper bounds.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input 'series' must be a pandas Series.")

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    print(f"Calculated IQR bounds for column '{series.name}': (Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f})")
    
    return lower_bound, upper_bound

def remove_outliers_by_bounds(df: pd.DataFrame, column: str, lower_bound: float, upper_bound: float) -> pd.DataFrame:
    """Removes rows from a DataFrame that are outside the given bounds for a specific column.

    Args:
        df: The pandas DataFrame to filter.
        column: The name of the column to check for outliers.
        lower_bound: The lower limit for the column value.
        upper_bound: The upper limit for the column value.

    Returns:
        A new DataFrame with the outliers removed.
    """
    initial_rows = len(df)
    
    filtered_df = df[
        (df[column] >= lower_bound) & (df[column] <= upper_bound)
    ].copy()
    
    rows_removed = initial_rows - len(filtered_df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} outliers from column '{column}' based on bounds ({lower_bound:.2f}, {upper_bound:.2f}).")
        
    return filtered_df