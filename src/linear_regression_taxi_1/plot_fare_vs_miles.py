import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Define paths relative to the project root, assuming the script is run from there.
artifacts_dir = Path("artifacts") / "linear_regression_taxi_1"
input_csv_path = artifacts_dir / "chicago_taxi_train.csv"
output_png_path = artifacts_dir / "fare_vs_miles.png"

# Ensure the artifacts directory exists
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Load the dataset
print(f"Loading data from {input_csv_path}...")
df = pd.read_csv(input_csv_path)

# Create the scatter plot
print("Creating plot...")
plt.figure(figsize=(10, 6))
plt.scatter(df['TRIP_MILES'], df['FARE'], alpha=0.5)

# Add labels and title
plt.xlabel("Trip Miles")
plt.ylabel("Fare")
plt.title("Fare vs. Trip Miles in Chicago Taxis")

# Save the plot as a PNG file
plt.savefig(output_png_path)

print(f"Plot saved as {output_png_path}")
