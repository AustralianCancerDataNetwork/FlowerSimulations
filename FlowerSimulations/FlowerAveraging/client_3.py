from collections import OrderedDict
import torch
import pandas as pd
import numpy as np
from flwr.client import NumPyClient, ClientApp

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, file_path):
        # Initialize with the path to the CSV file
        self.data = pd.read_csv(file_path)
    
    def get_parameters(self, config):
        # Not needed for this task, return an empty list or a default value
        return []
    
    def set_parameters(self, parameters):
        # Not applicable for CSV processing, but required for the interface
        pass

    def fit(self, parameters, config):
    # Calculate column averages
        column_averages = self.data.mean().values

        # Debugging: Print the calculated column averages
        print(f"Client {config.get('cid', 'unknown')} calculated column averages: {column_averages}")

        # Return parameters, dataset size, and column averages as individual float values
        return self.get_parameters(config={}), len(self.data), {f"column_{i}_average": avg for i, avg in enumerate(column_averages)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        column_averages = self.data.mean().values

        # Return column averages as individual float values
        return 0.0, len(self.data), {f"column_{i}_average": avg for i, avg in enumerate(column_averages)}



def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(file_path="data/data_3.csv").to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:5007",
        client=client_fn(cid="0"),
    )
