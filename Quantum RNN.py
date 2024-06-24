#Model 1
from google.colab import files
uploaded = files.upload()

!pip install pennylane

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pennylane as qml

# Read the csv file
data = pd.read_csv('co2_mm_mlo.csv')
print(data.head())

# Extract relevant columns
data_relevant = data[['year', 'month', 'average']]

# Handle missing values by forward filling
data_relevant['average'] = data_relevant['average'].replace(-9.99, None).ffill()

# Normalize the data
scaler = MinMaxScaler()
data_relevant['average_scaled'] = scaler.fit_transform(data_relevant[['average']])

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_train = X_train[:, :, None]
X_test = X_test[:, :, None]
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits)}
qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")

class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class HybridRNN(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, output_dim):
        super(HybridRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_qubits)
        self.quantum = QuantumLayer()
        self.fc_out = nn.Linear(n_qubits, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)  # Correctly shape h0 for batch processing
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.quantum(out)
        out = self.fc_out(out)
        return out

# Hyperparameters
sequence_length = 12
input_dim = 1
hidden_dim = 50
output_dim = 1

# Initialize model, loss function, and optimizer
model = HybridRNN(sequence_length, input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_test)
        val_loss = criterion(val_output, y_test)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = criterion(model(X_test), y_test)
print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions for 2023
model.eval()
last_sequence = torch.tensor(data_relevant['average_scaled'].values[-sequence_length:], dtype=torch.float32).view(1, -1, 1)
predictions = []

for _ in range(12):
    with torch.no_grad():
        next_value = model(last_sequence).item()
    predictions.append(next_value)
    last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor(next_value).view(1, 1, 1)), dim=1)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print('Predicted CO2 levels for 2023:', predictions.flatten())
#---------------------------------------------#
#Model 2 (with improvised predictions)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('co2_mm_mlo.csv')
print(data.head())

# Extract relevant columns and handle missing values
data_relevant = data[['year', 'month', 'average']]
data_relevant['average'] = data_relevant['average'].replace(-9.99, None).ffill()

# Normalize the data
scaler = MinMaxScaler()
data_relevant['average_scaled'] = scaler.fit_transform(data_relevant[['average']])

# Create sequences for RNN
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 12
X, y = create_sequences(data_relevant['average_scaled'].values, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Define the quantum circuit and device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits)}
qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")

class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class HybridRNN(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, output_dim):
        super(HybridRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, n_qubits)
        self.quantum = QuantumLayer()
        self.fc2 = nn.Linear(n_qubits, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.quantum(out)
        out = self.fc2(out)
        return out

# Hyperparameters
sequence_length = 12
input_dim = 1
hidden_dim = 50
output_dim = 1

# Initialize model, loss function, and optimizer
model = HybridRNN(sequence_length, input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Train the model
epochs = 1000
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_test)
        val_loss = criterion(val_output, y_test)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = criterion(model(X_test), y_test)
print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions for 2023
model.eval()
last_sequence = torch.tensor(data_relevant['average_scaled'].values[-sequence_length:], dtype=torch.float32).view(1, -1, 1).to(device)
predictions = []

for _ in range(12):
    with torch.no_grad():
        next_value = model(last_sequence).item()
    predictions.append(next_value)
    last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor(next_value).view(1, 1, 1).to(device)), dim=1)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print('Predicted CO2 levels for 2023:', predictions.flatten())

# Plot actual vs predicted CO2 levels
plt.figure(figsize=(12, 6))
plt.plot(data_relevant['year'] + data_relevant['month']/12, data_relevant['average'], label='Actual')
plt.plot(np.arange(2023, 2024, 1/12), predictions, label='Predicted')
plt.xlabel('Year')
plt.ylabel('CO2 Levels')
plt.title('CO2 Level Predictions for 2023')
plt.legend()
plt.show()

# Create a list of months for 2023
months = [f'2023-{i:02d}' for i in range(1, 13)]

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(months, classical_predictions, marker='o', linestyle='-', color='b')
plt.title('Predicted Monthly Average CO2 Levels for 2023 using QRNN')
plt.xlabel('Month')
plt.ylabel('CO2 Levels (ppm)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Actual CO2 levels for 2023
actual_values = np.array([419.00, 419.47, 420.31, 420.99, 423.36, 424.00, 423.68, 421.83, 419.68, 418.50, 418.82, 420.46])

# Create a list of months for 2023
months = [f'2023-{i:02d}' for i in range(1, 13)]

# Convert predictions and actual values to pandas DataFrames
classical_predictions_df = pd.DataFrame(classical_predictions, columns=['classical_predictions'])
qrnn_predictions_df = pd.DataFrame(predictions, columns=['qrnn_predictions'])
actual_values_df = pd.DataFrame(actual_values, columns=['actual_values'])

# Combine predictions and actual values into a single DataFrame
combined_predictions_df = pd.DataFrame(months, columns=['month'])
combined_predictions_df['classical_predictions'] = classical_predictions_df
combined_predictions_df['qrnn_predictions'] = qrnn_predictions_df
combined_predictions_df['actual_values'] = actual_values_df

# Plot the predictions and actual values for each month
plt.figure(figsize=(10, 6))
plt.plot(months, combined_predictions_df['classical_predictions'], marker='o', linestyle='-', color='b', label='RNN Predicted')
plt.plot(months, combined_predictions_df['qrnn_predictions'], marker='o', linestyle='-', color='r', label='QRNN Predicted')
plt.plot(months, combined_predictions_df['actual_values'], marker='o', linestyle='-', color='g', label='Actual')
plt.title('Predicted vs Actual Monthly Average CO2 Levels for 2023')
plt.xlabel('Month')
plt.ylabel('CO2 Levels (ppm)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print predictions
print('Predicted CO2 levels for 2023 using RNN:', classical_predictions.flatten())
print('Predicted CO2 levels for 2023 using QRNN:', qrnn_predictions.flatten())
print('Actual CO2 levels for 2023:', actual_values)

# Make predictions for 2023 and 2024
model.eval()
last_sequence = torch.tensor(data_relevant['average_scaled'].values[-sequence_length:], dtype=torch.float32).view(1, -1, 1).to(device)
predictions_qrnn = []

for _ in range(24):
    with torch.no_grad():
        next_value = model(last_sequence).item()
    predictions_qrnn.append(next_value)
    last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor(next_value).view(1, 1, 1).to(device)), dim=1)

# Convert predictions back to original scale
predictions_qrnn = scaler.inverse_transform(np.array(predictions_qrnn).reshape(-1, 1))
print('Predicted CO2 levels for 2023-2024:', predictions_qrnn.flatten())

# Actual CO2 levels for 2023
actual_values_2023 = np.array([419.00, 419.47, 420.31, 420.99, 423.36, 424.00, 423.68, 421.83, 419.68, 418.50, 418.82, 420.46])

# Create a list of months for 2023 and 2024
months = [f'2023-{i:02d}' for i in range(1, 13)] + [f'2024-{i:02d}' for i in range(1, 13)]

# Plot actual vs predicted CO2 levels
plt.figure(figsize=(12, 6))
plt.plot(data_relevant['year'] + data_relevant['month'] / 12, data_relevant['average'], label='Actual', color='black')
plt.plot(np.arange(2023, 2025, 1 / 12)[:24], np.append(actual_values_2023, predictions_qrnn.flatten()[:12]), label='QRNN Predicted', color='red')
plt.plot(np.arange(2023, 2025, 1 / 12)[:24], np.append(actual_values_2023, classical_predictions.flatten()[:12]), label='RNN Predicted', color='blue')
plt.xlabel('Year')
plt.ylabel('CO2 Levels (ppm)')
plt.title('Predicted Carbon emmission values for 2023 and 2024')
plt.legend()
plt.show()


