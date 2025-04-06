from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import pandas as pd

# Load client training data from CSV file
X_train = pd.read_csv('ver_data/data_3.csv')
X_train = X_train.to_numpy()  # Convert to NumPy array for easier manipulation

# Extract dataset dimensions
[trainval, feat] = X_train.shape  # trainval: Number of training samples, feat: Number of features

# Hyperparameters
learning_rate = 0.01  # Learning rate for weight updates
num_of_neurons = 5  # Number of neurons in the client's layer

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self):
        """Initialize the client with random weights and placeholders."""
        self.current_round = 0  # Track the current training round
        
        # Initialize weights for the first layer with small random values
        self.W1 = np.random.randn(num_of_neurons, feat) 

        # Placeholder for the previous round's gradient updates received from the server
        self.previous_dZ = np.zeros((num_of_neurons, trainval), dtype=np.float32)

        # Placeholder for the computed activation values
        self.Z1 = np.zeros((num_of_neurons, trainval), dtype=np.float32)

    def get_parameters(self, config):
        """Return the client's computed activation matrix (Z1)."""
        return [self.Z1]  # Send computed Z1 back to the server

    def set_parameters(self, parameters):
        """Extract and update client state with parameters received from the server."""
        if parameters:
            # Extract the server round number (scalar)
            self.current_round = parameters[0].item()
            
            # Extract the previous round's aggregated gradient matrix from the server
            self.previous_dZ = parameters[1]  

            print(f"Iteration No.: {self.current_round}")
            # print(f"Received Round {self.current_round} - Previous Avg:\n{self.previous_dZ}")

            # If there is any update in the received gradient, update weights accordingly
            if np.any(self.previous_dZ):
                dW1 = (1/trainval) * np.dot(self.previous_dZ, X_train)  # Compute weight update
                self.W1 = self.W1 - learning_rate * dW1  # Update weights using gradient descent

            # Compute the activation values for the next layer
            self.Z1 = np.dot(self.W1, X_train.T)  # Forward propagation step

    ########################################################

    def fit(self, parameters, config):
        """Handle the federated training process."""
        self.set_parameters(parameters)  # Process received parameters
        return self.get_parameters({}), 1, {}  # Return updated activations (Z1)

    def evaluate(self, parameters, config):
        """Evaluation function (not used in this case)."""
        return 0.0, 1, {}  # Returning dummy values as evaluation is not performed

# Function to create a new client instance
def client_fn(cid: str):
    """Create and return an instance of the FlowerClient."""
    return FlowerClient().to_client()

# Initialize Flower Client Application
app = ClientApp(client_fn=client_fn)

# Legacy execution mode (for standalone client execution)
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="115.146.84.72:5010",  # Address of the federated learning server
        client=client_fn(cid="0"),  # Initialize the client instance
    )
