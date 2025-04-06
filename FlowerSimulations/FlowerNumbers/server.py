from typing import List, Tuple, Dict, Optional
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Metrics, EvaluateIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

import numpy as np

class CustomFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_avg = None  # Now stores a matrix

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Send server_round and previous average matrix to clients"""
        server_round_array = np.array([server_round])
        
        # Initialize previous average matrix (3x3) if first round
        if self.previous_avg is None:
            self.previous_avg = np.zeros((3, 3), dtype=np.float32)
            
        parameters = ndarrays_to_parameters([server_round_array, self.previous_avg])
        
        clients = client_manager.sample(num_clients=3, min_num_clients=3)
        print(f"Round {server_round} - Sending server_round and avg matrix:\n{self.previous_avg}")
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        """Aggregate client matrices element-wise"""
        if not results:
            return None, {}
        
        # Extract client matrices
        client_matrices = [
            parameters_to_ndarrays(res.parameters)[0]  # Get first parameter (matrix)
            for _, res in results
        ]
        
        # Compute element-wise average
        avg_matrix = np.mean(client_matrices, axis=0)
        self.previous_avg = avg_matrix  # Store for next round
        
        print(f"Round {server_round} - Average Matrix:\n{avg_matrix}")
        return None, {"Accuracy": server_round}  # Store as metric

# Strategy and configuration
strategy = CustomFedAvg(min_available_clients=3)
config = ServerConfig(num_rounds=3)

# ServerApp
app = ServerApp(config=config, strategy=strategy)

# Legacy mode execution
if __name__ == "__main__":
    from flwr.server import start_server
    start_server(
        server_address="0.0.0.0:5007",
        config=config,
        strategy=strategy,
    )