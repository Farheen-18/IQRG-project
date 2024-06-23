#To upload the file from desktop
from google.colab import files 
uploaded = files.upload()

import pandas as pd
#Read the csv file
data = pd.read_csv('co2_mm_mlo.csv')
print(data.head())

# Display the first few rows of the data
data.head()

# Extract relevant columns
data_relevant = data[['year', 'month', 'average']]

# Handle missing values by forward filling
data_relevant['average'] = data_relevant['average'].replace(-9.99, None).ffill()

# Normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_relevant['average_scaled'] = scaler.fit_transform(data_relevant[['average']])

# Display the first few rows of the preprocessed data
data_relevant.head()

import numpy as np
from sklearn.model_selection import train_test_split

# Create sequences of data for RNN
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Define sequence length
sequence_length = 12  # Using 12 months to predict the next month

# Prepare the data
X, y = create_sequences(data_relevant['average_scaled'].values, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Define the RNN architecture
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Reshape the data for the RNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions for 2023
import numpy as np

# Create input sequence from the last 12 months in the training data
last_sequence = data_relevant['average_scaled'].values[-sequence_length:]
input_sequence = last_sequence.reshape((1, sequence_length, 1))

# Predict the next 12 months
predictions = []
for _ in range(12):
    next_value = model.predict(input_sequence)[0][0]
    predictions.append(next_value)
    input_sequence = np.append(input_sequence[:, 1:, :], [[[next_value]]], axis=1)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Print predictions
print('Predicted CO2 levels for 2023:', predictions.flatten())
