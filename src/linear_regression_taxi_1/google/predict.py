# predict_fare.py

# Load required modules
import pandas as pd
import keras
import numpy as np
import pickle # Import pickle to load the settings

# Import utils to allow unpickling of the ExperimentSettings object.
# This replaces the implicit dependency on the 'google-ml-edu' package.

# Load the trained model
# Note: Ensure 'taxi_fare_model.keras' is in the same directory or provide the full path
try:
    model = keras.models.load_model('taxi_fare_model.keras')
    print("Trained model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'taxi_fare_model.keras' is in the correct location.")
    exit() # Exit if the model cannot be loaded

# Load the model settings
try:
    with open('model_settings.pkl', 'rb') as f:
        settings = pickle.load(f)
    print("Model settings loaded successfully.")
except Exception as e:
    print(f"Error loading model settings: {e}")
    ### # You might need to manually define settings if the file is missing
    ### # For example:
    ### # from ml_edu.experiment import ExperimentSettings
    ### # settings = ExperimentSettings(input_features=['TRIP_MILES', 'TRIP_MINUTES'])
    # If the file is missing, you might need to retrain the model
    # or manually define the settings object using the new utils module:
    # settings = utils.ExperimentSettings(input_features=['TRIP_MILES', 'TRIP_MINUTES'], ...)
    exit() # Exit if settings cannot be loaded


# Define functions to make predictions (copied from the notebook)
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare_from_model(model, df, features, label=None, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [],
          features[0]: [], features[1]: [] if len(features) > 1 else []}

  if label and label in batch.columns:
      data["OBSERVED_FARE"] = []
      data["L1_LOSS"] = []

  for i in range(batch_size):
    predicted = predicted_values[i][0]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data[features[0]].append(batch.at[i, features[0]])
    if len(features) > 1:
        data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

    if label and label in batch.columns:
        observed = batch.at[i, label]
        data["OBSERVED_FARE"].append(format_currency(observed))
        data["L1_LOSS"].append(format_currency(abs(observed - predicted)))


  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

# Example usage: Make predictions on a sample of the original training data
# In a real scenario, you would load new data here.
chicago_taxi_dataset_for_prediction = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
prediction_df = chicago_taxi_dataset_for_prediction.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE')] # Include FARE to show comparison

# Ensure TRIP_MINUTES is in the prediction data if it was used for training
if 'TRIP_MINUTES' in settings.input_features and 'TRIP_MINUTES' not in prediction_df.columns:
     prediction_df['TRIP_MINUTES'] = prediction_df['TRIP_SECONDS']/60


# Make predictions
# Pass 'FARE' as the label name if you want to see the comparison (L1_LOSS)
output = predict_fare_from_model(model, prediction_df, settings.input_features, label='FARE', batch_size=50)
show_predictions(output)

# Example of making predictions on new data (without the FARE column)
# new_data = pd.DataFrame({
#     'TRIP_MILES': [5.0, 10.0, 1.5],
#     'TRIP_SECONDS': [900, 1800, 300]
# })
# if 'TRIP_MINUTES' in settings.input_features:
#     new_data['TRIP_MINUTES'] = new_data['TRIP_SECONDS'] / 60
#
# output_new = predict_fare_from_model(model, new_data, settings.input_features, batch_size=len(new_data))
# print("\nPredictions on new data:")
# show_predictions(output_new)