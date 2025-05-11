from typing import List, Tuple, Dict, Optional
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Metrics, EvaluateIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from flwr.common import GetPropertiesIns, Parameters

import numpy as np

nx = np.array([2, 5, 5, 1])  # Layer sizes

class CustomFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Needs to adjust this size for a different dataset
        self.dZ1 = np.zeros((5, 245057), dtype=np.float32) 

        # Needs to adjust this size for a different dataset
        self.v_client_Z = np.zeros((5, 245057), dtype=np.float32)
         
        self.h_client_Zs = {}
        self.client_names = {} # Key: UUID (cid), Value: client_name

        # Initialize weights and biases for each layer
        self.W1_h = np.random.randn(nx[1], nx[0])
        
        self.W2 = np.random.randn(nx[2], nx[1])
        self.b2 = np.random.randn(nx[2], 1)

        self.W3 = np.random.randn(nx[3], nx[2])
        self.b3 = np.random.randn(nx[3], 1)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):

        print("--------------------------------------------------------------")
        # This is a function (configure_fit) used to SEND data to participating clients

        # Round Number in an array (to make it compatible with the send
        # data fromat)

        server_round_array = np.array([server_round])

        print("ROUND (configure_fit): "+str(server_round))

        '''
        if self.dZ1 is None:
            self.dZ1 = np.zeros((3, 3), dtype=np.float32) 

        if self.A1 is None:
            self.A1 = np.zeros((3, 3), dtype=np.float32)
        '''

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

            custom_avg = np.zeros((3, 3), dtype=np.float32)
            
            # Check if the current federated round is even
            if server_round % 2 == 0:
            
                # Case: Sending full Z1 matrix to the vertical client (client_V)
                if client_name == "client_V":
                    # Convert Z1 matrix to float32 NumPy array for compatibility
                    custom_avg = np.array(self.v_client_Z).astype(np.float32)
                    # Package the round number and Z1 matrix into Flower Parameters
                    parameters = ndarrays_to_parameters([server_round_array.astype(np.float32), custom_avg])
                    print(f"\033[95m[Round {server_round}] Sending Z1 (same) matrix to {client_name}:\033[0m")
                
                # Case: Sending first part of Z1_V to horizontal client H_1 (first 400 rows)
                elif client_name == "client_H_1":
                    # Convert slice to float32 NumPy array
                    custom_avg = np.array(self.v_client_Z[:, 0:10000]).astype(np.float32)
                    # Package round number and sliced Z1 matrix
                    parameters = ndarrays_to_parameters([server_round_array.astype(np.float32), custom_avg])
                    print(f"\033[95m[Round {server_round}] Sending part of Z1_V matrix to {client_name}:\033[0m")
                
                # Case: Sending second part of Z1_V to horizontal client H_2 (rows 400 to 1372)
                elif client_name == "client_H_2":
                    # Convert slice to float32 NumPy array
                    custom_avg = np.array(self.v_client_Z[:,10000:245057]).astype(np.float32)
                    # Package round number and sliced Z1 matrix
                    parameters = ndarrays_to_parameters([server_round_array.astype(np.float32), custom_avg])
                    print(f"\033[95m[Round {server_round}] Sending part of Z1_V matrix to {client_name}:\033[0m")

            else:
                # For odd number of rounds,
                # The server will send the model parameters to the H-clients
                # and will sned dZ1 (in full) to V-client 
                if client_name == "client_V":
                    # Log the action for transparency in the training process
                    print(f"\033[95m[Round {server_round}] Sending dZ1 matrix to {client_name}:\033[0m")
                    
                    # Convert the dZ1 gradient matrix (from the backward pass) to float32 format
                    custom_avg = np.array(self.dZ1).astype(np.float32)
                    
                    # Package the current round number and dZ1 matrix into Flower-compatible parameters
                    parameters = ndarrays_to_parameters([server_round_array.astype(np.float32), custom_avg])

                # For horizontal clients (e.g., client_H_1, client_H_2)
                else:
                    # Log the action of sending full model parameters for horizontal clients
                    print(f"\033[95m[Round {server_round}] Sending model parameters to H-clients:\033[0m")
                    
                    # Convert and send the model parameters (weights and biases) needed for local training
                    parameters = ndarrays_to_parameters([
                        server_round_array.astype(np.float32),  # Round number
                        self.W1_h.astype(np.float32),          # Horizontal client's first layer weights
                        self.W2.astype(np.float32),            # Shared second layer weights
                        self.b2.astype(np.float32),            # Shared second layer biases
                        self.W3.astype(np.float32),            # Output layer weights
                        self.b3.astype(np.float32),            # Output layer biases
                    ])


            # Convert variables from arrays to parameters to send them to
            # the participating clients
            # parameters = ndarrays_to_parameters([server_round_array, custom_avg])
        
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

        # This loop will iterate for number of participating clients,
        # i.e., it will iteratre for three times if there are
        # three participating clients
         
        for client_proxy, res in results:
            # Each item in `results` is a tuple: (client_proxy, FitRes)
            # You can access things like metrics, parameters, etc., from the FitRes

            # Loop through each result in the results list
            # Each item is a tuple: (client_proxy, res)
            
            # Extract the parameters sent by the client (serialized format)
            serialized_params = res.parameters

            metrics = res.metrics

            # Get the client's name from the metrics dictionary
            client_name = metrics['client_name']

            # Convert the serialized parameters to NumPy arrays
            # This returns a list of arrays; for example: [matrix], where matrix is (3, 3)
            
            ndarray_list = parameters_to_ndarrays(serialized_params)

            if server_round % 2 == 0:
                if client_name == "client_H_1":
                    # For client_H_1 (received concatenated matrix from client_H_1)
                    matrix_all = ndarray_list[0]  # The received combined matrix

                    # Define the column lengths for each parameter: [W1_h, W2, b2, W3, b3, dZ1]
                    lengths = [2, 5, 1, 1, 1, 10000]  # Total 410 columns

                    # ---- Unpack W1_h_1 ----
                    start = 0
                    end = lengths[0]
                    W1_h_1 = matrix_all[:, start:end]  # Shape: (5, 2)

                    # ---- Unpack W2_h_1 ----
                    start = end
                    end += lengths[1]
                    W2_h_1 = matrix_all[:, start:end]  # Shape: (5, 5)

                    # ---- Unpack b2_h_1 ----
                    start = end
                    end += lengths[2]
                    b2_h_1 = matrix_all[:, start:end]  # Shape: (5, 1)

                    # ---- Unpack W3_h_1 ----
                    start = end
                    end += lengths[3]
                    W3_h_1 = matrix_all[:, start:end]  # Shape: (5, 1)
                    W3_h_1 = W3_h_1.reshape([1, 5])    # Reshape to original shape: (1, 5)

                    # ---- Unpack b3_h_1 ----
                    start = end
                    end += lengths[4]
                    b3_h_1 = matrix_all[:, start:end]  # Shape: (5, 1)
                    b3_h_1 = b3_h_1[0]                 # Extract first row (original b3 shape: (1,))

                    # ---- Unpack dZ1 ----
                    start = end
                    end += lengths[5]

                    self.dZ1[:, 0:10000] = matrix_all[:, start:end]  # Fill first 400 columns of dZ1

                elif client_name == "client_H_2":

                    # For client_H_2 (received concatenated matrix from client_H_2)
                    matrix_all = ndarray_list[0]  # The received combined matrix

                    # Define the column lengths for each parameter: [W1_h, W2, b2, W3, b3, dZ1]
                    lengths = [2, 5, 1, 1, 1, 235057]  # Total 982 columns

                    # ---- Unpack W1_h_2 ----
                    start = 0
                    end = lengths[0]
                    W1_h_2 = matrix_all[:, start:end]  # Shape: (5, 2)

                    # ---- Unpack W2_h_2 ----
                    start = end
                    end += lengths[1]
                    W2_h_2 = matrix_all[:, start:end]  # Shape: (5, 5)

                    # ---- Unpack b2_h_2 ----
                    start = end
                    end += lengths[2]
                    b2_h_2 = matrix_all[:, start:end]  # Shape: (5, 1)

                    # ---- Unpack W3_h_2 ----
                    start = end
                    end += lengths[3]
                    W3_h_2 = matrix_all[:, start:end]  # Shape: (5, 1)
                    W3_h_2 = W3_h_2.reshape([1, 5])    # Reshape to original shape: (1, 5)

                    # ---- Unpack b3_h_2 ----
                    start = end
                    end += lengths[4]
                    b3_h_2 = matrix_all[:, start:end]  # Shape: (5, 1)
                    b3_h_2 = b3_h_2[0]                 # Extract first row (original b3 shape: (1,))

                    # ---- Unpack dZ1 ----
                    start = end
                    end += lengths[5]

                    self.dZ1[:, 10000:245057] = matrix_all[:, start:end]  # Fill next 972 columns of dZ1

            else:
                # Get the first matrix from the list
                matrix = ndarray_list[0]
                # Store matrix based on client type
                if client_name == "client_V":
                    self.v_client_Z = matrix
        
        if server_round % 2 == 0:
            self.W1_h = (W1_h_1 + W1_h_2)/2
            
            self.W2 = (W2_h_1 + W2_h_2)/2
            self.b2 = (b2_h_1 + b2_h_2)/2

            self.W3 = (W3_h_1 + W3_h_2)/2
            self.b3 = (b3_h_1 + b3_h_2)/2
            
        return None, {"Accuracy": server_round}
        
# Strategy and configuration
strategy = CustomFedAvg(min_available_clients=3)
config = ServerConfig(num_rounds=3000)

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