import time

start_time = time.time()

from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import Config, Scalar

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import pandas as pd

import matplotlib.pyplot as plt

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

##############################################################################
# Machine Learning (Neural Network) model supplementary materials

def sigmoid(z):
    """Compute the sigmoid activation function."""
    sig = 1 / (1 + np.exp(-z))
    return sig

# Load and preprocess server-side dataset

folder = "skin"

v_client_data = pd.read_csv(f'datasets/{folder}/data_V.csv')
v_client_data = v_client_data.to_numpy()

# Read the first mask from the CSV file
mask_client_V = pd.read_csv(f'masks/{folder}/mask_client_V.csv')

# Convert the DataFrame to a NumPy array (matrix)
mask_client_V = mask_client_V.to_numpy()

# Extract features (X_train) and labels (y_train)
X_train = v_client_data[:, 0]
y_train = v_client_data[:, 1]

X_train = np.array(X_train).reshape(-1, 1)

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

learning_rate = 0.5  # Learning rate for weight updates

##############################################################################

# Define a Flower client by subclassing NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self):
        # Set a unique human-readable client name (used by server for identification)
        self.client_name = "client_V"

        self.A1_from_server = None  # Store A1 received from the server

        self.Z1_v = np.zeros([nx[1], trainval])

        self.dZ1 = np.zeros([nx[1], trainval])

        # Initialize weights and biases for each layer
        self.W1_v = np.random.randn(nx[1], nx[0])
        self.b1 = np.random.randn(nx[1], 1)

        self.W2 = np.random.randn(nx[2], nx[1])
        self.b2 = np.random.randn(nx[2], 1)

        self.W3 = np.random.randn(nx[3], nx[2])
        self.b3 = np.random.randn(nx[3], 1)

        self.loss_history = []
        self.accuracy_history = []

        self.current_round = 1

    def get_properties(self, config: Config) -> dict[str, Scalar]:

        # Sends properties to the server when requested.
        # Used by the server to identify the client before the first round.
        # Returns a dictionary that includes the human-readable name of the client.

        return {
            "client_name": self.client_name,
        }

    def get_parameters(self, config):

        print("ROUND  (get_parameters, SEND): "+str(self.current_round))

        matrix_to_send = np.zeros((3, 3), dtype=np.float32)
            

        # The first four rounds are left to implement the secure communication
        # channels between the clients via the server
        if self.current_round > 4:

            if self.current_round % 2 == 1:
                # Calculation of Z1 matrix at the client side
                self.Z1_v = np.dot(self.W1_v, X_train.T) + self.b1

                # For odd rounds, return self.data + my_a
                matrix_to_send = np.array(self.Z1_v + mask_client_V, dtype=np.float32)

                print("\033[95mZ1 send to the server\033[0m")

            else:
                # For even rounds

                # Split dZ1 into two parts
                dZ1_1 = self.dZ1[:, 0:10000]
                dZ1_2 = self.dZ1[:, 10000:245057]

                # Step 1: Convert to bytes
                dZ1_1 = dZ1_1.astype(np.float32)
 
                dZ1_1_bytes = dZ1_1.tobytes()

                # Step 2: Pad the bytes (AES block size = 16 bytes)
                dZ1_1_padded_bytes = pad(dZ1_1_bytes, AES.block_size)

                # Step 3: Load the AES key
                with open("aes_key.bin", "rb") as key_file:
                    key = key_file.read() 

                # Step 4: Encrypt
                cipher = AES.new(key, AES.MODE_ECB)
                dZ1_1_encrypted = cipher.encrypt(dZ1_1_padded_bytes)

                ###################################

                # Step 1: Convert to bytes
                dZ1_2 = dZ1_2.astype(np.float32)
 
                dZ1_2_bytes = dZ1_2.tobytes()

                # Step 2: Pad the bytes (AES block size = 16 bytes)
                dZ1_2_padded_bytes = pad(dZ1_2_bytes, AES.block_size)

                # Step 4: Encrypt
                cipher = AES.new(key, AES.MODE_ECB)
                dZ1_2_encrypted = cipher.encrypt(dZ1_2_padded_bytes)

                ###################################

                matrix_to_send = np.frombuffer(dZ1_1_encrypted+dZ1_2_encrypted, dtype=np.uint8)
                print("\033[95mdZ1 send to the server\033[0m")

            # print("Matrix send: "+str(matrix_to_send))

        return [matrix_to_send]

    def set_parameters(self, parameters):
        
        print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))
        
        if parameters:
            # Extract the current round number (assumed to be a scalar)
            self.current_round = parameters[0].item()

            print("--------------------------------------------------------------")

            if self.current_round > 4:

                if self.current_round % 2 == 0:

                    print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))

                    # Extract the average matrix (assumed to be a NumPy array)
                    self.A1_from_server = parameters[1]

                    print("\033[94mReceived A1 from server\033[0m")

                    # Forward Propagation

                    Z2 = np.dot(self.W2, self.A1_from_server) + self.b2
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
                    dW2 = (1 / trainval) * np.dot(dZ2, self.A1_from_server.T)
                    db2 = (1 / trainval) * np.sum(dZ2, axis=1, keepdims=True)

                    dA1 = self.A1_from_server * (1 - self.A1_from_server)
                    dZ1 = np.dot(self.W2.T, dZ2) * dA1  # Server will send dZ1 back to clients
                    dW1_v = (1 / trainval) * np.dot(dZ1, X_train)
                    db1 = (1 / trainval) * np.sum(dZ1, axis=1, keepdims=True)

                    # Update weights using gradient descent
                    self.W1_v = self.W1_v - learning_rate * dW1_v
                    self.b1 = self.b1 - learning_rate * db1

                    self.W2 = self.W2 - learning_rate * dW2
                    self.b2 = self.b2 - learning_rate * db2

                    self.W3 = self.W3 - learning_rate * dW3
                    self.b3 = self.b3 - learning_rate * db3

                    # Store the latest gradient to be transmitted to the H-clients (via server)
                    self.dZ1 = dZ1

                    # print("for testing purpose, dZ1: "+str(self.dZ1))

                    # Compute training accuracy
                    pred_train = A3 > 0.5
                    train_accuracy = 1 - np.sum(abs(pred_train - y_train)) / trainval

                    # Store the loss and accuracy
                    self.loss_history.append(Loss)
                    self.accuracy_history.append(train_accuracy)

                    print(f"\033[92mIteration {self.current_round} - Train Accuracy: {train_accuracy:.4f} - Train Loss: {Loss:.4f}\033[0m")

                    if self.current_round  == 2000:
                        self.save_metrics()

                else:
                    # do not do anything 
                    pass

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
        df.to_csv(f'results/{folder}/training_metrics.csv', index=False)

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label="Loss", color="red")
        plt.xlabel("Training Rounds")
        plt.ylabel("Loss")
        plt.title("Training Loss over Rounds")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/loss_plot.jpeg', dpi=300)

         # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history, label="Accuracy", color="blue")
        plt.xlabel("Training Rounds")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy over Rounds")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/accuracy_plot.jpeg', dpi=300)

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

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")