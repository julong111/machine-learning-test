# train_model.py

# Load required modules
import pandas as pd
import keras
### import ml_edu.experiment
### import ml_edu.results
from src.linear_regression_taxi_1 import utils
from matplotlib import pyplot as plt
import pickle # Import pickle to save the trained model

# Load the dataset
chicago_taxi_dataset = pd.read_csv("../chicago_taxi_train.csv")

# Update the dataframe and create TRIP_MINUTES
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
# In a local script, you might not display the head directly, but save to a file or print
# print(training_df.head(200))


# Define functions to build and train a model
def create_model(
    ### settings: ml_edu.experiment.ExperimentSettings,
    settings: utils.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple linear regression model."""
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    ### settings: ml_edu.experiment.ExperimentSettings,
    settings: utils.ExperimentSettings,
### ) -> ml_edu.experiment.Experiment:
) -> utils.Experiment:
  """Train the model by feeding it data."""

  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  ### return ml_edu.experiment.Experiment(
  return utils.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

print("SUCCESS: defining linear regression functions complete.")

# Train the model with two features (TRIP_MILES and TRIP_MINUTES)
### settings = ml_edu.experiment.ExperimentSettings(
settings = utils.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model = create_model(settings, metrics)

experiment = train_model('two_features', model, training_df, 'FARE', settings)

# Save the trained model
model.save('taxi_fare_model.keras') # Save in Keras native format
print("\nTrained model saved as 'taxi_fare_model.keras'")

# Optionally save the settings for inference
with open('model_settings.pkl', 'wb') as f:
    pickle.dump(settings, f)
print("Model settings saved as 'model_settings.pkl'")

# You might also want to save plots to files in a local script
### ml_edu.results.plot_experiment_metrics(experiment, ['rmse'])
utils.plot_experiment_metrics(experiment, ['rmse'])
plt.savefig('training_rmse_plot.png')
plt.clf() # Clear the current figure

# Note: plot_model_predictions for the 3D plot is interactive and might require
# special handling or skipping in a non-interactive script.
# If you need a static image, you might need to use different plotting code
# or ensure plotly/kaleido is set up correctly for static export.
# For simplicity, we'll skip saving this plot in the training script.