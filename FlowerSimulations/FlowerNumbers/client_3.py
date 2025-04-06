from flwr.client import NumPyClient, ClientApp
import numpy as np

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

my_a = 3
# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self):
        # Initialize with a 3x3 matrix (e.g., all 30s)
        self.data = np.full((3, 3), 30, dtype=np.float32)
        self.current_round = 0
        self.previous_avg = np.zeros((3, 3), dtype=np.float32)

    def get_parameters(self, config):
        """Return the client's 3x3 matrix"""
        return [self.data + my_a]

    def set_parameters(self, parameters):
        """Extract server_round and previous average matrix"""
        if parameters:
            self.current_round = parameters[0].item()  # Server round (scalar)
            self.previous_avg = parameters[1]         # Previous avg (matrix)
            
            # Update client data: add previous average element-wise
            self.data = self.data + self.previous_avg
            print(f"Received Round {self.current_round} - Previous Avg:\n{self.previous_avg}")

    ########################################################

    def fit(self, parameters, config):
        """Required method (process parameters)"""
        self.set_parameters(parameters)
        return self.get_parameters({}), 1, {}  # Return client data

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
        server_address="115.146.84.72:5010",
        client=client_fn(cid="0"),
    )
