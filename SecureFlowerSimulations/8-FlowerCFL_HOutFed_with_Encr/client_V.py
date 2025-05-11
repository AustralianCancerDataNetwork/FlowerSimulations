import time

start_time = time.time()

from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import Config, Scalar

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import pandas as pd

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

##############################################################################
# Machine Learning (Neural Network) model supplementary materials

# Load and preprocess server-side dataset

folder = "skin"

X_train = pd.read_csv(f'datasets/{folder}/data_V.csv')
X_train = X_train.to_numpy()

X_train = np.array(X_train).reshape(-1, 1)

# Get training data size and feature count
[trainval, feat] = X_train.shape

learning_rate = 0.5  # Learning rate for weight updates

##############################################################################

# Define a Flower client by subclassing NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self):
        # Set a unique human-readable client name (used by server for identification)
        self.client_name = "client_V"

        # Needs to adjust this size for a different dataset

        # NEEDS TO CHANGE HERE

        self.Z1_v = np.zeros([5, 245057])

        # Needs to adjust this size for a different dataset

        # NEEDS TO CHANGE HERE

        self.dZ1 = np.zeros([5, 245057])

        # Initialize weights and biases for each layer
        self.W1_v = np.random.randn(5, feat)
        self.b1 = np.random.randn(5, 1)

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

        print("ROUND  (get_parameters, SEND): "+str(self.current_round))

        matrix_to_send = np.zeros((3, 3), dtype=np.float32)

        if self.current_round > 4:
            if self.current_round % 2 == 1:
                # Calculation of Z1 matrix at the client side
                self.Z1_v = np.dot(self.W1_v, X_train.T) + self.b1

                # Split Z1_v into two parts

                # NEEDS TO CHANGE HERE

                Z1_v_1 = self.Z1_v[:, 0:10000]
                Z1_v_2 = self.Z1_v[:, 10000:245057]

                # Step 1: Convert to bytes
                Z1_v_1 = Z1_v_1.astype(np.float32)
 
                Z1_v_1_bytes = Z1_v_1.tobytes()

                # Step 2: Pad the bytes (AES block size = 16 bytes)
                Z1_v_1_padded_bytes = pad(Z1_v_1_bytes, AES.block_size)

                # Step 3: Load the AES key
                with open("aes_key.bin", "rb") as key_file:
                    key = key_file.read() 

                # Step 4: Encrypt

                cipher = AES.new(key, AES.MODE_ECB)
                Z1_v_1_encrypted = cipher.encrypt(Z1_v_1_padded_bytes)

                ###################################

                # Step 1: Convert to bytes
                Z1_v_2 = Z1_v_2.astype(np.float32)
 
                Z1_v_2_bytes = Z1_v_2.tobytes()

                # Step 2: Pad the bytes (AES block size = 16 bytes)
                Z1_v_2_padded_bytes = pad(Z1_v_2_bytes, AES.block_size)

                # Step 4: Encrypt
                cipher = AES.new(key, AES.MODE_ECB)
                Z1_v_2_encrypted = cipher.encrypt(Z1_v_2_padded_bytes)

                ###################################

                # For odd rounds, return self.data + my_a
                matrix_to_send = np.frombuffer(Z1_v_1_encrypted+Z1_v_2_encrypted, dtype=np.uint8)

                print("\033[95mZ1 send to the server\033[0m")

            else:
                # For even rounds, return self.data only (or anything else you choose)

                matrix_to_send = np.array(self.dZ1, dtype=np.float32)
                print("\033[95mdZ1 send to the server\033[0m")
            
        # print("Matrix send: "+str(matrix_to_send))

        return [matrix_to_send]

    def set_parameters(self, parameters):

        # This method is used to RECEIVE data from the server
        
        print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))
        
        if parameters:
            # Extract the current round number (assumed to be a scalar)
            self.current_round = parameters[0].item()

            print("--------------------------------------------------------------")

            if self.current_round > 4:
                if self.current_round % 2 == 0:

                    print("ROUND  (set_parameters, RECEIVE): "+str(self.current_round))

                    print("\033[94mReceived Z1 (do nothing) from server\033[0m")

                    '''
                    # Extract the average matrix (assumed to be a NumPy array)
                    
                    '''

                else:
                    print("\033[94mReceived dZ1 from server\033[0m")

                    # Extract the average matrix (assumed to be a NumPy array)
                    self.dZ1 = parameters[1]

                    if not np.all(self.dZ1 == 0):

                        dW1 = (1/trainval) * np.dot(self.dZ1, X_train)  # Compute weight update
                        self.W1_v = self.W1_v - learning_rate * dW1  # Update weights using gradient descent

                        # Compute the activation values for the next layer
                        self.Z1 = np.dot(self.W1_v, X_train.T)  # Forward propagation step

                # need to do do backward propagation
            

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

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")