from typing import List, Tuple, Dict, Optional
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Metrics, EvaluateIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from flwr.common import GetPropertiesIns, Parameters

import numpy as np

def sigmoid(z):
    """Compute the sigmoid activation function."""
    sig = 1 / (1 + np.exp(-z))
    return sig

class CustomFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dZ1 = None  # Now stores a matrix
        self.A1 = None
        self.v_client_Z = None 
        self.h_client_Zs = {}
        self.client_names = {} # Key: UUID (cid), Value: client_name

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):

        print("--------------------------------------------------------------")
        # This is a function (configure_fit) used to SEND data to participating clients

        # Round Number in an array (to make it compatible with the send
        # data fromat)

        server_round_array = np.array([server_round])

        print("ROUND (configure_fit): "+str(server_round))

        if self.dZ1 is None:
            self.dZ1 = np.zeros((3, 3), dtype=np.float32) 

        if self.A1 is None:
            self.A1 = np.zeros((3, 3), dtype=np.float32)

        clients = client_manager.sample(num_clients=3, min_num_clients=3)

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
        
        fit_instructions = []

  
        for client in clients:

            # Extract the unique identifier (UUID string) of the client
            
            client_id = client.cid  

            # Get the human-readable name of the client using the UUID
            # If the client ID is not found in the dictionary, default to "Unknown"
            
            client_name = self.client_names.get(client_id, "Unknown")

            # Determine which matrix to send based on the current round number
            # In even-numbered rounds, 'client_V' receives A1 and other clients receives
            # a sero matrix 
            # In odd-numbered rounds, all clients including V-client receives dZ1
            # sent by the V-client
            
            if server_round % 2 == 0:

                if client_name == "client_V":
                    custom_avg = self.A1
                    print(f"\033[95m[Round {server_round}] Sending A1 (nonlinear activations) matrix to {client_name}:\033[0m")
                else:
                    custom_avg = np.zeros((3, 3), dtype=np.float32)
            else:
                custom_avg = self.dZ1

            # Convert variables from arrays to parameters to send them to
            # the participating clients
            parameters = ndarrays_to_parameters([server_round_array, custom_avg])
        
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

        print("ROUND (aggregate_fit): "+str(server_round))
        
        if not results:
           # If no clients returned results (e.g., due to failure or dropout),
           # return None for parameters and an empty dictionary for metrics 
           return None, {}
        
        # Initialize an empty list to hold the matrices received from clients
        client_matrices = []

        for client_proxy, res in results:
            # Each item in `results` is a tuple: (client_proxy, FitRes)
            # You can access things like metrics, parameters, etc., from the FitRes

            # Loop through each result in the results list
            # Each item is a tuple: (client_proxy, res)
            
            # Extract the parameters sent by the client (serialized format)
            serialized_params = res.parameters

            # Convert the serialized parameters to NumPy arrays
            # This returns a list of arrays; for example: [matrix], where matrix is (3, 3)
            
            ndarray_list = parameters_to_ndarrays(serialized_params)

            # Get the first matrix from the list
            matrix = ndarray_list[0]

            # Append the matrix to our client_matrices list
            client_matrices.append(matrix)

            metrics = res.metrics
            if "client_name" in metrics:
                # If client has sent a custom metric called "client_name", print it
                
                print(f"\033[94mReceived result from: {metrics['client_name']}\033[0m")

            # Get the client's name from the metrics dictionary
            client_name = metrics['client_name']

            # Store matrix based on client type
            if client_name == "client_V":
                if server_round % 2 == 1:
                    self.v_client_Z = matrix
                else:
                    self.dZ1 = matrix
            else:
                self.h_client_Zs[client_name] = matrix

        if server_round % 2 == 1:
            # Compute the element-wise sum of all client matrices
            # axis=0 means we are summing each corresponding element across all matrices
        
            all_Z_matrices = np.sum(client_matrices, axis=0)
            self.A1 = sigmoid(all_Z_matrices)

        return None, {"Accuracy": server_round}
        
# Strategy and configuration
strategy = CustomFedAvg(min_available_clients=3)
config = ServerConfig(num_rounds=1000)

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