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

# Dry Run for firs 8 rounds

## Round 1-4

These rounds will be used to setup the secure communication channels between the clients via the server. At the moment, these are empty (their implementation is pending)

## Round 5

SERVER sends dZ1 (it will be zeros in the first round) to H-clients

SERVER sends nothing to V-client

H-CLIENTS receives dZ1 (zeros)

V-CLIENT sends Z1 to the server

H-CLIENTs sends Z1 to the server (with masks)

SERVER receives Z1 from all clients (with masks)

SERVER will calculate the sum of all Zs

SERVER will calculate nonlinear activation of Zs, A1

## Round 6

SERVER sends A1 to V-client

V-CLIENT receives A1

V-CLIENT does local training

V-CLIENT sends dZ1 to the server (to send it to the H-clients, in encrypted form)

SERVER received dZ1 from V-client

## Round 7

SERVER sends dZ1 (received from V-client, in encrypted form) to H-clients

SERVER sends nothing to V-client

H-CLIENTS receives dZ1

H-CLIENTS decrypts dZ1

H-CLIENTS update their model parameters

V-CLIENT sends Z1 to the server (with masks)

H-CLIENTs sends Z1 to the server (with masks)

SERVER receives Z1 from all clients

SERVER will calculate the sum of all Zs

SERVER will calculate nonlinear activation of Zs, A1

## Round 8

SERVER sends A1 to V-client

V-CLIENT receives A1

V-CLIENT does local training

V-CLIENT sends dZ1 to the server (to send it to the H-clients, in encrypted form)

SERVER received dZ1 from V-client