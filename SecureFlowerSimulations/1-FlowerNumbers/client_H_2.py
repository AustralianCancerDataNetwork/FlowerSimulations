from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import Config, Scalar

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Custom variable used to modify data returned by get_parameters
my_a = 1

# Define a Flower client by subclassing NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self):
        # Set a unique human-readable client name (used by server for identification)
        self.client_name = "client_H_2"

        # Initialize the client’s local data matrix (e.g., filled with 10s)
        # This 3x3 matrix represents the client's model or custom state
        self.data = np.full((3, 3), 10, dtype=np.float32)

        # Variable to keep track of the current global round
        self.current_round = 0

        # Placeholder for the previous average matrix received from the server
        self.previous_avg = np.zeros((3, 3), dtype=np.float32)

    def get_properties(self, config: Config) -> dict[str, Scalar]:

        # Sends properties to the server when requested.
        # Used by the server to identify the client before the first round.
        # Returns a dictionary that includes the human-readable name of the client.
        
        return {
            "client_name": self.client_name,
        }

    def get_parameters(self, config):
        
        # This method is used to SEND data to the server
        
        # Sends the client's parameters to the server.
        # This method is called when the server wants to fetch the client's current state.
        # It returns the `self.data` matrix plus `my_a`, which allows for customized behavior.
        
        return [self.data + my_a]

    def set_parameters(self, parameters):
        
        # This method is used to RECEIVE data from the server
       
        # Receives parameters from the server before training.
        # The server sends a list where:
        # - The first element is the server round number (scalar)
        # - The second element is the previous average matrix from the last round
        # This method updates the client's state based on those values.
        
        if parameters:
            # Extract the current round number (assumed to be a scalar)
            self.current_round = parameters[0].item()

            # Extract the average matrix (assumed to be a NumPy array)
            self.previous_avg = parameters[1]

            # Update the local data matrix by adding the average matrix element-wise
            self.data = self.data + self.previous_avg

            # Display debug information to track what was received and used
            print(f"\033[34mReceived Round {self.current_round} - Previous Avg:\n{self.previous_avg}\033[0m")

    ########################################################

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"Fitting {self.client_name} in round {self.current_round}")
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
