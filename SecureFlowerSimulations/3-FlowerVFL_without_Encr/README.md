# Flower Client-Server Setup for Vertical Federated Learning

This implementation demonstrates a vertical federated learning setup using the Flower framework. 

The clients only communicate with the server. If the clients want to communicate with other clients, they have to do this via the server.

For further implementation details, refer to the code's inline comments.

## Data Distribution

There are three clients:

client_H_1: holds input features only

client_H_2: holds input features only

client_V: holds input and output features

## Note

Please make sure to align the data points of each client before training.

# Dry Run for four rounds

## Round 1

SERVER sends dZ1 (it will be zeros in the first round) to H-clients

SERVER sends nothing to V-client

H-CLIENTS receives dZ1 (zeros)

V-CLIENT sends Z1 to the server

H-CLIENTs sends Z1 to the server

SERVER receives Z1 from all clients

SERVER will calculate the sum of all Zs

SERVER will calculate nonlinear activation of Zs, A1

## Round 2

SERVER sends A1 to V-client

V-CLIENT receives A1

V-CLIENT does local training

V-CLIENT sends dZ1 to the server (to send it to the H-clients)

SERVER received dZ1 from V-client

## Round 3

SERVER sends dZ1 (received from V-client) to H-clients

SERVER sends nothing to V-client

H-CLIENTS receives dZ1

H-CLIENTS update their model parameters

V-CLIENT sends Z1 to the server

H-CLIENTs sends Z1 to the server

SERVER receives Z1 from all clients

SERVER will calculate the sum of all Zs

SERVER will calculate nonlinear activation of Zs, A1

## Round 4

SERVER sends A1 to V-client

V-CLIENT receives A1

V-CLIENT does local training

V-CLIENT sends dZ1 to the server (to send it to the H-clients)

SERVER received dZ1 from V-client