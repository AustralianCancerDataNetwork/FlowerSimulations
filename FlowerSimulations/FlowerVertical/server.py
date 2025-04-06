from typing import List, Tuple, Dict, Optional
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Metrics, EvaluateIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def sigmoid(z):
    """Compute the sigmoid activation function."""
    sig = 1 / (1 + np.exp(-z))
    return sig

# Load and preprocess server-side dataset
server_data = pd.read_csv('ver_data/data_s.csv')
server_data = server_data.to_numpy()

# Extract features (X_train) and labels (y_train)
X_train = server_data[:, 0:1]
y_train = server_data[:, 1]

# Get training data size and feature count
[trainval, feat] = X_train.shape

# Define network architecture
L = 4  # Number of layers (including input and output layers)
nx = np.array([X_train.shape[1], 5, 5, 1])  # Layer sizes

# Training settings
num_of_iteraitons = 500  # Total training rounds
Loss = np.zeros((num_of_iteraitons, 1))  # Loss array
train_accuracy = np.zeros((num_of_iteraitons, 1))  # Accuracy tracking

# Initialize layer activation matrices
Z1_s = np.zeros([nx[1], trainval])
Z1 = np.zeros([nx[1], trainval])
A1 = np.zeros([nx[1], trainval])

Z2 = np.zeros([nx[2], trainval])
A2 = np.zeros([nx[2], trainval])

Z3 = np.zeros([nx[3], trainval])
A3 = np.zeros([nx[3], trainval])

learning_rate = 0.5  # Learning rate for weight updates

# Custom federated averaging strategy
class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        """Initialize the custom FedAvg strategy with additional weight parameters."""
        super().__init__(**kwargs)
        self.previous_dZ = None  # Store previous updates for consistency

        # Initialize weights and biases for each layer
        self.W1_s = np.random.randn(nx[1], nx[0]) * 0.01
        self.b1 = np.random.randn(nx[1], 1)

        self.W2 = np.random.randn(nx[2], nx[1]) * 0.01
        self.b2 = np.random.randn(nx[2], 1)

        self.W3 = np.random.randn(nx[3], nx[2]) * 0.01
        self.b3 = np.random.randn(nx[3], 1)

        self.loss_history = []
        self.accuracy_history = []

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Send the current training round and previous aggregated updates to clients."""
        server_round_array = np.array([server_round])

        # Check if `self.previous_dZ` has been initialized
        # This only happens in the first round of training, as `self.previous_dZ`
        # starts as `None` before the first aggregation.
        if self.previous_dZ is None:
            
            # Initialize `self.previous_dZ` as a zero matrix with dimensions (nx[1], trainval)
            # - `nx[1]` represents the number of neurons in the first hidden layer
            # - `trainval` represents the total number of training examples
            # This matrix will be used to store the previous round's gradient updates (`dZ1`)
            # and will be sent to the clients in the next round for federated updates.
            self.previous_dZ = np.zeros((nx[1], trainval), dtype=np.float32)
            
            # `dtype=np.float32` ensures memory efficiency and faster computation compared to
            # the default `float64` type, which is often unnecessary for neural network training.

        
        # Convert to Flower parameters format
        parameters = ndarrays_to_parameters([server_round_array, self.previous_dZ])
        
        # Select clients for training
        clients = client_manager.sample(num_clients=3, min_num_clients=3)
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        """Aggregate client updates and perform forward & backward propagation."""
        if not results:
            return None, {}
        
        # Extract matrices received from clients
        client_Z_matrices = [
            parameters_to_ndarrays(res.parameters)[0]  # Extract the first parameter (update matrix)
            for _, res in results
        ]
        
        # Compute average across all client updates
        total_client_matrices = np.sum(client_Z_matrices, axis=0)

        # Forward propagation
        Z1_s = np.dot(self.W1_s, X_train.T) + self.b1
        Z1 = Z1_s + total_client_matrices  # Incorporate client updates
        A1 = sigmoid(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = sigmoid(Z3)

        # Compute loss using binary cross-entropy
        Loss = (-1 / trainval) * np.sum(y_train * np.log(A3) + (1 - y_train) * np.log(1 - A3))

        # Backward propagation
        dZ3 = A3 - y_train
        dW3 = (1 / trainval) * np.dot(dZ3, A2.T)
        db3 = (1 / trainval) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = A2 * (1 - A2)
        dZ2 = np.dot(self.W3.T, dZ3) * dA2
        dW2 = (1 / trainval) * np.dot(dZ2, A1.T)
        db2 = (1 / trainval) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = A1 * (1 - A1)
        dZ1 = np.dot(self.W2.T, dZ2) * dA1  # Server will send dZ1 back to clients
        dW1_s = (1 / trainval) * np.dot(dZ1, X_train)
        db1 = (1 / trainval) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights using gradient descent
        self.W1_s -= learning_rate * dW1_s
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

        # Store the latest gradient for the next round
        self.previous_dZ = dZ1

        # Compute training accuracy
        pred_train = A3 > 0.5
        train_accuracy = 1 - np.sum(abs(pred_train - y_train)) / trainval

         # Store the loss and accuracy
        self.loss_history.append(Loss)
        self.accuracy_history.append(train_accuracy)

        print(f"Iteration {server_round} - Train Accuracy: {train_accuracy:.4f}")
        return 1, {"Accuracy": train_accuracy, "Loss": Loss}  # Return accuracy as a metric
    
    def save_metrics(self):
        """Save Loss and Accuracy values to a CSV file."""
        df = pd.DataFrame({"Round": np.arange(len(self.loss_history)), 
                           "Loss": self.loss_history, 
                           "Accuracy": self.accuracy_history})
        df.to_csv("results/training_metrics.csv", index=False)

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_of_iteraitons), self.loss_history, label="Loss", color="red")
        plt.xlabel("Training Rounds")
        plt.ylabel("Loss")
        plt.title("Training Loss over Rounds")
        plt.legend()
        plt.grid()
        plt.savefig("results/loss_plot.jpeg", dpi=300)

         # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_of_iteraitons), self.accuracy_history, label="Accuracy", color="blue")
        plt.xlabel("Training Rounds")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy over Rounds")
        plt.legend()
        plt.grid()
        plt.savefig("results/accuracy_plot.jpeg", dpi=300)

# Define the federated learning strategy
strategy = CustomFedAvg(min_available_clients=3)
config = ServerConfig(num_rounds=num_of_iteraitons)

# Initialize the Flower server application
app = ServerApp(config=config, strategy=strategy)

# Start the federated server
if __name__ == "__main__":
    from flwr.server import start_server
    start_server(
        server_address="0.0.0.0:5010",  # Listen on port 5007
        config=config,
        strategy=strategy,
    )


strategy.save_metrics()