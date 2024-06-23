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



