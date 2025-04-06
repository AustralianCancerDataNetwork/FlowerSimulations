# Vertical Federated Learning using Flower Tool
This repository contains the implementation of a Vertical Federated Learning (VFL) system using the Flower framework. The system consists of a server and multiple clients that collaboratively train a neural network model without sharing their raw data. Below is a detailed explanation of the technical aspects of the server and client implementations.

## Introduction
Vertical Federated Learning (VFL) is a privacy-preserving machine learning technique where different parties (clients) hold different features of the same dataset. The Flower framework facilitates the implementation of federated learning systems by providing tools for communication between the server and clients.

# Server Implementation
Data Loading and Preprocessing
The server loads its dataset from a CSV file (data_s.csv). The dataset is preprocessed by converting it into a NumPy array for easier manipulation. The features (X_train) and labels (y_train) are extracted from the dataset.

```python
import pandas as pd

# Load server dataset from CSV file
server_data = pd.read_csv('ver_data/data_s.csv')

# Convert DataFrame to NumPy array
server_data = server_data.to_numpy()

# Extract input features (first column) and target labels (second column)
X_train = server_data[:, 0:1]
y_train = server_data[:, 1]
```

## Network Architecture
The neural network consists of 4 layers (including input and output layers). The layer sizes are defined as follows:

```python
L = 4  # Number of layers
nx = np.array([X_train.shape[1], 5, 5, 1])  # Layer sizes
```

## Custom Federated Averaging Strategy
The CustomFedAvg class extends the FedAvg strategy provided by Flower. It initializes weights and biases for each layer and implements the federated averaging logic.

```python
class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_dZ = None
        self.W1_s = np.random.randn(nx[1], nx[0]) * 0.01
        self.b1 = np.random.randn(nx[1], 1)
        self.W2 = np.random.randn(nx[2], nx[1]) * 0.01
        self.b2 = np.random.randn(nx[2], 1)
        self.W3 = np.random.randn(nx[3], nx[2]) * 0.01
        self.b3 = np.random.randn(nx[3], 1)
        self.loss_history = []
        self.accuracy_history = []
```

## Training Process
The server orchestrates the training process by sending the current training round and previous aggregated updates to the clients. It then aggregates the client updates and performs forward and backward propagation.

```python
def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
    server_round_array = np.array([server_round])
    if self.previous_dZ is None:
        self.previous_dZ = np.zeros((nx[1], trainval), dtype=np.float32)
    parameters = ndarrays_to_parameters([server_round_array, self.previous_dZ])
    clients = client_manager.sample(num_clients=3, min_num_clients=3)
    return [(client, FitIns(parameters, {})) for client in clients]

def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
    client_Z_matrices = [parameters_to_ndarrays(res.parameters)[0] for _, res in results]
    total_client_matrices = np.sum(client_Z_matrices, axis=0)
    Z1_s = np.dot(self.W1_s, X_train.T) + self.b1
    Z1 = Z1_s + total_client_matrices
    A1 = sigmoid(Z1)
    Z2 = np.dot(self.W2, A1) + self.b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(self.W3, A2) + self.b3
    A3 = sigmoid(Z3)
    Loss = (-1 / trainval) * np.sum(y_train * np.log(A3) + (1 - y_train) * np.log(1 - A3))
    dZ3 = A3 - y_train
    dW3 = (1 / trainval) * np.dot(dZ3, A2.T)
    db3 = (1 / trainval) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = A2 * (1 - A2)
    dZ2 = np.dot(self.W3.T, dZ3) * dA2
    dW2 = (1 / trainval) * np.dot(dZ2, A1.T)
    db2 = (1 / trainval) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = A1 * (1 - A1)
    dZ1 = np.dot(self.W2.T, dZ2) * dA1
    dW1_s = (1 / trainval) * np.dot(dZ1, X_train)
    db1 = (1 / trainval) * np.sum(dZ1, axis=1, keepdims=True)
    self.W1_s -= learning_rate * dW1_s
    self.b1 -= learning_rate * db1
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
    self.W3 -= learning_rate * dW3
    self.b3 -= learning_rate * db3
    self.previous_dZ = dZ1
    pred_train = A3 > 0.5
    train_accuracy = 1 - np.sum(abs(pred_train - y_train)) / trainval
    self.loss_history.append(Loss)
    self.accuracy_history.append(train_accuracy)
    print(f"Iteration {server_round} - Train Accuracy: {train_accuracy:.4f}")
    return 1, {"Accuracy": train_accuracy, "Loss": Loss}
```

## Metrics and Visualization
The server saves the training metrics (loss and accuracy) to a CSV file and generates plots to visualize the training progress.

```python
def save_metrics(self):
    df = pd.DataFrame({"Round": np.arange(len(self.loss_history)), 
                       "Loss": self.loss_history, 
                       "Accuracy": self.accuracy_history})
    df.to_csv("results/training_metrics.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_of_iteraitons), self.loss_history, label="Loss", color="red")
    plt.xlabel("Training Rounds")
    plt.ylabel("Loss")
    plt.title("Training Loss over Rounds")
    plt.legend()
    plt.grid()
    plt.savefig("results/loss_plot.jpeg", dpi=300)
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_of_iteraitons), self.accuracy_history, label="Accuracy", color="blue")
    plt.xlabel("Training Rounds")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy over Rounds")
    plt.legend()
    plt.grid()
    plt.savefig("results/accuracy_plot.jpeg", dpi=300)
```

# Client Implementation
## Data Loading
Each client loads its dataset from a CSV file. The dataset is converted into a NumPy array for easier manipulation.

```python
X_train = pd.read_csv('ver_data/data_3.csv')
X_train = X_train.to_numpy()
[trainval, feat] = X_train.shape
```

## Client Initialization
The FlowerClient class initializes the client with random weights and placeholders for the previous round's gradient updates and computed linear part (Z1) values.

```python
class FlowerClient(NumPyClient):
    def __init__(self):
        self.current_round = 0
        self.W1 = np.random.randn(num_of_neurons, feat)
        self.previous_dZ = np.zeros((num_of_neurons, trainval), dtype=np.float32)
        self.Z1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)
```

## Parameter Handling
The client handles parameters received from the server, updates its weights, and computes the activation values for the next layer.

```python
def set_parameters(self, parameters):
    if parameters:
        self.current_round = parameters[0].item()
        self.previous_dZ = parameters[1]
        if np.any(self.previous_dZ):
            dW1 = (1/trainval) * np.dot(self.previous_dZ, X_train)
            self.W1 = self.W1 - learning_rate * dW1
        self.Z1 = np.dot(self.W1, X_train.T)
```

# Running the System
To run the system, start the server and clients in separate terminal windows.

## Start the Server:
```python
python server.py
```

## Start the Client:
```python
python client_1.py
python client_2.py
python client_3.py
```

Ensure that the server and clients are running simultaneously and that the server address and port are correctly configured.