from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import Config, Scalar

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import pandas as pd

import matplotlib.pyplot as plt

##############################################################################
# Machine Learning (Neural Network) model supplementary materials

# Load client training data from CSV file
folder = "skin"

h_client_1_data = pd.read_csv(f'datasets/{folder}/data_H_2.csv')
h_client_1_data = h_client_1_data.to_numpy()

# Extract features (X_train) and labels (y_train)
X_train = h_client_1_data[:, 0:2]
y_train = h_client_1_data[:, 2]

# Get training data size and feature count
[trainval, feat] = X_train.shape

# Define network architecture
L = 4  # Number of layers (including input and output layers)
nx = np.array([X_train.shape[1], 5, 5, 1])  # Layer sizes

# Initialize layer activation matrices
A1 = np.zeros([nx[1], trainval])

Z2 = np.zeros([nx[2], trainval])
A2 = np.zeros([nx[2], trainval])

Z3 = np.zeros([nx[3], trainval])
A3 = np.zeros([nx[3], trainval])

# Hyperparameters
learning_rate = 0.5  # Learning rate for weight updates
num_of_neurons = 5  # Number of neurons in the client's layer

def sigmoid(z):
    """Compute the sigmoid activation function."""
    sig = 1 / (1 + np.exp(-z))
    return sig

##############################################################################

# Define a Flower client by subclassing NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self):
        # Set a unique human-readable client name (used by server for identification)
        self.client_name = "client_H_2"

        # Initialize weights and biases for each layer
        self.W1_h = np.zeros((nx[1], nx[0]), dtype=np.float32)
        
        self.W2 = np.zeros((nx[2], nx[1]), dtype=np.float32)
        self.b2 = np.zeros((nx[2], 1), dtype=np.float32)

        self.W3 = np.zeros((nx[3], nx[2]), dtype=np.float32)
        self.b3 = np.zeros((nx[3], 1), dtype=np.float32)

        # Placeholder for the gradient updates received from the server
        self.dZ1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)

        # Placeholder for the computed activation values
        self.Z1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)

        # Placeholder for the computed activation values
        self.Z1_v = np.zeros((num_of_neurons, trainval), dtype=np.float32)

        self.loss_history = []
        self.accuracy_history = []

        # Variable to keep track of the current global round
        self.current_round = 1

    def get_properties(self, config: Config) -> dict[str, Scalar]:

        # Sends properties to the server when requested.
        # Used by the server to identify the client before the first round.
        # Returns a dictionary that includes the human-readable name of the client.
        
        return {
            "client_name": self.client_name,
        }

    def get_parameters(self, config):

        # This method is used to SEND data to the server
        
        # This method is called when the server requests parameters from the client.
        # It sends different data depending on whether the round is odd or even.

        print("ROUND  (get_parameters, SEND): "+str(self.current_round))
        
        # Doing the same thing for both rounds
        # it could be avoided, but just to demonstrate
        # there are alternative rounds

        matrix_to_send = np.zeros((3, 3), dtype=np.float32)
        
        if self.current_round % 2 == 1:
            self.Z1 = np.dot(self.W1_h, X_train.T)
        else:
            print("\033[94mReceived Z1_v from server\033[0m")

            self.Z1 = self.Z1 + self.Z1_v
            A1 = sigmoid(self.Z1)

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
            dW1_h = (1 / trainval) * np.dot(dZ1, X_train)

            # Update weights using gradient descent
            self.W1_h = self.W1_h - learning_rate * dW1_h

            self.W2 = self.W2 - learning_rate * dW2
            self.b2 = self.b2 - learning_rate * db2

            self.W3 = self.W3 - learning_rate * dW3
            self.b3 = self.b3 - learning_rate * db3

            # Store the latest gradient to be transmitted to the H-clients (via server)
            self.dZ1 = dZ1

            def reshape_to_5_rows(matrix):
                flat = matrix.flatten()
                needed_cols = int(np.ceil(len(flat) / 5))
                padded = np.pad(flat, (0, needed_cols * 5 - len(flat)))
                reshaped = padded.reshape(5, -1)
                return reshaped
            
            # Reshape and concatenate
            W1_h_r = reshape_to_5_rows(self.W1_h)
            W2_r = reshape_to_5_rows(self.W2)
            b2_r = reshape_to_5_rows(self.b2)
            W3_r = reshape_to_5_rows(self.W3)
            b3_r = reshape_to_5_rows(self.b3)

            matrix_to_send = np.concatenate([W1_h_r, W2_r, b2_r, W3_r, b3_r, self.dZ1], axis=1)

            # Compute training accuracy
            pred_train = A3 > 0.5
            train_accuracy = 1 - np.sum(abs(pred_train - y_train)) / trainval

            # Store the loss and accuracy
            self.loss_history.append(Loss)
            self.accuracy_history.append(train_accuracy)

            print(f"\033[92mIteration {self.current_round} - Train Accuracy: {train_accuracy:.4f} - Train Loss: {Loss:.4f}\033[0m")

            if self.current_round  == 3000:
                self.save_metrics()
        
        return [matrix_to_send]

    def set_parameters(self, parameters):

        print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))
        
        # This method is used to RECEIVE data from the server
       
        # Receives parameters from the server before training.
        # The server sends a list where:
        # - The first element is the server round number (scalar)
        # - The second element is the previous average matrix from the last round
        # This method updates the client's state based on those values.
        
        if parameters:
            print("--------------------------------------------------------------")

            # Extract the current round number (assumed to be a scalar)
            self.current_round = parameters[0].item()

            if self.current_round % 2 == 1:

                print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))

                print("\033[94mReceived model parameters from server\033[0m")
 
                self.W1_h = parameters[1]
                self.W2 = parameters[2]
                self.b2 = parameters[3]
                self.W3 = parameters[4]
                self.b3 = parameters[5]

            else:
                print("\033[94mReceived Z1_V from server\033[0m")
                self.Z1_v = parameters[1]

                
    ########################################################

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # print(f"Fitting {self.client_name} in round {self.current_round}")
        return self.get_parameters({}), 1, {"client_name": self.client_name}

    def evaluate(self, parameters, config):
        """Required method (unused)"""
        return 0.0, 1, {}  # Can be empty
    
    def save_metrics(self):
        """Save Loss and Accuracy values to a CSV file."""
        df = pd.DataFrame({"Round": np.arange(len(self.loss_history)), 
                           "Loss": self.loss_history, 
                           "Accuracy": self.accuracy_history})
        df.to_csv(f'results/{folder}/training_metrics_Client_H_2.csv', index=False)

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label="Loss", color="red")
        plt.xlabel("Training Rounds")
        plt.ylabel("Loss")
        plt.title("Training Loss over Rounds-Client_H_2")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/loss_plot_Client_H_2.jpeg', dpi=300)

         # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history, label="Accuracy", color="blue")
        plt.xlabel("Training Rounds")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy over Rounds-Client_H_2")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/accuracy_plot_Client_H_2.jpeg', dpi=300)

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:5010",
        client=client_fn(cid="0"),
    )
