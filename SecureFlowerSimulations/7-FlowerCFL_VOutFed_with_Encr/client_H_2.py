from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import Config, Scalar

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import pandas as pd

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Load client training data from CSV file
folder = "skin"

X_train = pd.read_csv(f'datasets/{folder}/data_H_2.csv')
X_train = X_train.to_numpy()  # Convert to NumPy array for easier manipulation

# Read the first mask from the CSV file
mask_client_H_1 = pd.read_csv(f'masks/{folder}/mask_client_H_2.csv')

# Convert the DataFrame to a NumPy array (matrix)
mask_client_H_1 = mask_client_H_1.to_numpy()

# Extract dataset dimensions
[trainval, feat] = X_train.shape  # trainval: Number of training samples, feat: Number of features

# Hyperparameters
learning_rate = 0.5  # Learning rate for weight updates
num_of_neurons = 5  # Number of neurons in the client's layer

# Define a Flower client by subclassing NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self):
        # Set a unique human-readable client name (used by server for identification)
        self.client_name = "client_H_2"

        # Initialize weights for the first layer with small random values
        self.W1 = np.random.randn(num_of_neurons, feat) * 0.01

        # Placeholder for the gradient updates received from the server
        self.dZ1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)

        # Placeholder for the computed activation values
        self.Z1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)

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
        
        if self.current_round > 4:
            if self.current_round % 2 == 1:
                matrix_to_send = np.dot(self.W1, X_train.T) + mask_client_H_1
                print("\033[95mZ1 send to the server\033[0m")
            else:
                matrix_to_send = np.dot(self.W1, X_train.T)

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

            if self.current_round > 6:
                if self.current_round % 2 == 1:

                    print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))
                    # Extract the average matrix (assumed to be a NumPy array)
                    encrypted_dZ1 = parameters[1]

                    encrypted_dZ1 = encrypted_dZ1.tobytes()

                    # Step 1: Load the AES key
                    with open("aes_key.bin", "rb") as key_file:
                        key = key_file.read()
                    
                    # Step 2: Decrypt
                    decipher = AES.new(key, AES.MODE_ECB)
                    decrypted_dZ1_padded = decipher.decrypt(encrypted_dZ1)

                    # Step 3: Unpad and convert back to matrix
                    decrypted_bytes = unpad(decrypted_dZ1_padded, AES.block_size)
                    self.dZ1 = np.frombuffer(decrypted_bytes, dtype=np.float32).reshape(mask_client_H_1.shape)

                    print("\033[94mReceived dZ1 from server\033[0m")
                    dW1 = (1/trainval) * np.dot(self.dZ1, X_train)  # Compute weight update
                    self.W1 = self.W1 - learning_rate * dW1  # Update weights using gradient descent

                    # Compute the activation values for the next layer
                    self.Z1 = np.dot(self.W1, X_train.T)  # Forward propagation step
                else:
                    pass

    ########################################################

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # print(f"Fitting {self.client_name} in round {self.current_round}")
        return self.get_parameters({}), 1, {"client_name": self.client_name}

    def evaluate(self, parameters, config):
        """Required method (unused)"""
        return 0.0, 1, {}  # Can be empty

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
