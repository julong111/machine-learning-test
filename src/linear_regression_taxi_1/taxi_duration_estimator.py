# -*- coding: utf-8 -*-
"""
Defines the DurationEstimator class for the taxi fare prediction project.

This module contains the DurationEstimator, which uses a pre-calculated
lookup table to estimate trip duration based on distance. This is a key
component of the feature engineering for the 'smart_model'.
"""

import json
from pathlib import Path


class DurationEstimator:
    """Loads a lookup table to provide trip duration estimates based on distance."""

    def __init__(self, config_path: Path):
        """Initializes the estimator by loading the lookup table."""
        print(f"Loading duration estimation rules from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                self.lookup_table = {float(k): v for k, v in json.load(f).items()}
            self.sorted_upper_bounds = sorted(self.lookup_table.keys())
            print("Duration estimator initialized successfully.")
        except FileNotFoundError:
            print(f"Error: Lookup table not found at {config_path}")
            print("Please run 'analyze_time_distance.py' first to generate the lookup table.")
            raise

    def estimate(self, miles: float) -> float:
        """Estimates trip duration by finding the correct bin in the loaded lookup table."""
        for upper_bound in self.sorted_upper_bounds:
            if miles < upper_bound:
                return self.lookup_table[upper_bound]
        # If miles is greater than the largest bin, use the value for the largest bin.
        return self.lookup_table[self.sorted_upper_bounds[-1]]
