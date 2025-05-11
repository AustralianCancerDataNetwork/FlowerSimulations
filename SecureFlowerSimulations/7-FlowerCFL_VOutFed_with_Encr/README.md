# Flower Client-Server Setup for Combined Federated Learning-VOutFed

This implementation demonstrates a combined federated learning setup using the Flower framework. 

The clients only communicate with the server. If the clients want to communicate with other clients, they have to do this via the server.

For further implementation details, refer to the code's inline comments.

## Data Distribution

There are three clients:

client_H_1: holds two input features for first 400 data points

client_H_2: holds the same two input features as client_H_1 but for the next 972 data points

client_V: holds the rest of two input features, along with the output feature for all 1372 data points (first 400 of client_H_1 and the next 972 for the client_H_2)

## Note

Please make sure to align the data points of each client before training.

Please make sure to adjust the number of data points allocated for each H-client in server.py file, in if elif block, in lines: 106-111

Also, in general, if the number of H-clients varies (from 2), then modify the server.py accordingly.

# Dry Run for firs 8 rounds

## Round 1-4

These rounds will be used to setup the secure communication channels between the clients via the server. At the moment, these are empty (their implementation is pending)

## Round 5

SERVER sends dZ1 (it will be zeros in the first round) to H-clients

SERVER sends nothing to V-client

H-CLIENTS receives dZ1 (zeros)

V-CLIENT sends Z1 to the server (with masks)

H-CLIENTs sends Z1 to the server (with masks)

SERVER receives Z1 from all clients 

SERVER concatenates Z1 matrices received from the H-clients

SERVER will calculate the sum of the concatenated Z and the Z1 matrix received from the V-client

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

SERVER concatenates Z1 matrices received from the H-clients

SERVER will calculate the sum of the concatenated Z and the Z1 matrix received from the V-client

SERVER will calculate nonlinear activation of Zs, A1

## Round 8

SERVER sends A1 to V-client

V-CLIENT receives A1

V-CLIENT does local training

V-CLIENT sends dZ1 to the server (to send it to the H-clients, in encrypted form)

SERVER received dZ1 from V-client