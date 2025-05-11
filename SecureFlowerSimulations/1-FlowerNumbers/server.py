from typing import List, Tuple, Dict, Optional
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Metrics, EvaluateIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from flwr.common import GetPropertiesIns, Parameters

import numpy as np

class CustomFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_avg = None  # Now stores a matrix
        self.client_names = {} # Key: UUID (cid), Value: client_name

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        # This is a function (configure_fit) used to SEND data to participating clients

        # Round Number in an array (to make it compatible with the send
        # data fromat)
        server_round_array = np.array([server_round])
        
        # Initialize previous average matrix (3x3), only in the first round
        # (as in the first round, this matrix will be zero as initialized in
        # __init__)
        if self.previous_avg is None:
            self.previous_avg = np.zeros((3, 3), dtype=np.float32)
            
        # Sample 3 clients from the pool of connected clients.
        # This ensures that at least 3 clients are available (min_num_clients),
        # and returns a list of 3 randomly selected clients for the current round.
        clients = client_manager.sample(num_clients=3, min_num_clients=3)

        # Get client properties before sending instructions
        for client in clients:
            # Create the GetPropertiesIns object to request client properties
            # - The 'config' dictionary can contain custom config parameters 
            # you want to send to the client.
            # - Here, we leave it empty since we only want to retrieve client properties (e.g., client_name).
            get_properties_ins = GetPropertiesIns(config={})

            # Send the get_properties request to the client
            # - timeout=5: Wait up to 5 seconds for the client to respond
            # - group_id=None: No specific group identifier used (can be useful in advanced setups)
            # - This triggers the client's `get_properties` method and returns a GetPropertiesRes object
            props_res = client.get_properties(get_properties_ins, timeout=5, group_id=None)

            # Extract the client name from the properties dictionary returned by the client
            # - If the client did not send a 'client_name', default to "Unknown"
            client_name = props_res.properties.get("client_name", "Unknown")

            # Store the client name using its client ID (UUID) as the key
            # - This allows us to identify and reference clients by their human-readable names in future rounds
            self.client_names[client.cid] = client_name

            # print(f"Client connected: {client_name} (cid={client.cid})")

        # Printing a simple meesage at Server to display the current
        # round number and the matrix to be transmitted
        # The matrix will have zeros for the first round
        # print(f"Round {server_round} - Sending server_round and avg matrix:\n{self.previous_avg}")
        
        # Create an empty list to hold client-instruction pairs (in this case
        # it is the round number and average matrix)
        fit_instructions = []

        # Loop over each client in the list of sampled clients
        for client in clients:

            # Extract the unique identifier (UUID string) of the client
            client_id = client.cid  

            # Get the human-readable name of the client using the UUID
            # If the client ID is not found in the dictionary, default to "Unknown"
            client_name = self.client_names.get(client_id, "Unknown")

            # Check if the current client is the special client 'client_V'
            if client_name == "client_V":
                # If yes, send a customized average matrix (e.g., twice the average)
                custom_avg = 2 * self.previous_avg
            else:
                # For all other clients, send the regular average matrix
                custom_avg = self.previous_avg

            
            # Convert variables from arrays to parameters to send them to
            # the participating clients
            parameters = ndarrays_to_parameters([server_round_array, custom_avg])
        

            # Print a message showing which client is about to receive fit instructions
            # print(f"Preparing fit instruction for client: {client_name} (cid={client_id})")
            
            # Create a FitIns object for this client
            # It includes:
            #   - parameters: the model parameters ()
            #   - config: an empty dictionary for optional training settings
            # In HFL, it is used to send may be hyper-parameters or something else,
            # here, it is an empty dictionary
            fit_instruction = FitIns(parameters, {})
            
            # Create a tuple of (client, fit_instruction)
            client_instruction_pair = (client, fit_instruction)
            
            # Add the tuple to the list
            fit_instructions.append(client_instruction_pair)

        # Return the list of (client, FitIns) pairs
        # Each pair specifies which instruction to send to which client
        # Flower uses this to send the FitIns (with parameters + config) to 
        # each corresponding client
        # In this case, all clients receive the same parameters, but the 
        # mapping ensures each client gets its own message
        return fit_instructions
        
    def aggregate_fit(self, server_round, results, failures):
        # This is a function (aggregate_fit) used to RECEIVE data from participating clients
        # This method is called by the server AFTER it receives the fit() results from clients.
        # It's responsible for:
        #   - Aggregating the updated parameters (here, 3x3 matrices) from all clients
        #   - Updating the server's state (e.g., storing the average matrix)
        #   - Returning optional aggregated metrics (like accuracy)
        
        if not results:
            # If no clients returned results (e.g., due to failure or dropout),
            # return None for parameters and an empty dictionary for metrics
            return None, {}
        
        # Initialize an empty list to hold the matrices received from clients
        client_matrices = []

        for client_proxy, res in results:
            # Each item in `results` is a tuple: (client_proxy, FitRes)
            # You can access things like metrics, parameters, etc., from the FitRes

            metrics = res.metrics
            if "client_name" in metrics:
                # If client has sent a custom metric called "client_name", print it
                # Also print the client's unique ID (cid)
                print(f"\033[94mReceived result from: {metrics['client_name']}\033[0m")

            # Loop through each result in the results list
            # Each item is a tuple: (client_proxy, res)
            
            # Extract the parameters sent by the client (serialized format)
            serialized_params = res.parameters

            # Convert the serialized parameters to NumPy arrays
            # This returns a list of arrays; for example: [matrix], where matrix is (3, 3)
            ndarray_list = parameters_to_ndarrays(serialized_params)

            # Get the first (and only) matrix from the list
            matrix = ndarray_list[0]

            # Append the matrix to our client_matrices list
            client_matrices.append(matrix)
    
        # Compute the element-wise average of all client matrices
        # Each client sends a (3x3) matrix; this computes the average across all clients
        # axis=0 means we are averaging each corresponding element across all matrices
        avg_matrix = np.mean(client_matrices, axis=0)

        # Store the computed average matrix to be sent to clients in the next round
        self.previous_avg = avg_matrix

        # Print the averaged matrix for the current round
        print(f"Round {server_round} - Average Matrix:\n{avg_matrix}")

        # Return values from aggregate_fit:
        # - First value is usually updated global model parameters (not used here, so we return None)
        # - Second is a dictionary of custom metrics (we return the round number as a fake "accuracy" metric)
        return None, {"Accuracy": server_round}
        

# Strategy and configuration
strategy = CustomFedAvg(min_available_clients=3)
config = ServerConfig(num_rounds=3)

# ServerApp
app = ServerApp(config=config, strategy=strategy)

# Legacy mode execution
if __name__ == "__main__":
    from flwr.server import start_server
    start_server(
        server_address="0.0.0.0:5010",
        config=config,
        strategy=strategy,
    )