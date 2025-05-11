# Flower Client-Server Setup for Combined Federated Learning-HOutFed

This implementation demonstrates a combined federated learning setup using the Flower framework. 

The clients only communicate with the server. If the clients want to communicate with other clients, they have to do this via the server.

For further implementation details, refer to the code's inline comments.

## Data Distribution

There are three clients:

client_H_1: holds two input features for first 400 data points, along with the output feature for these first 400 data points

client_H_2: holds the same two input features as client_H_1 but for the next 972 data points, along with the output feature for these first 972 data points

client_V: holds the rest of two input features only

## Note

Please make sure to align the data points of each client before training.

Please make sure to adjust the number of data points allocated for each H-client in server.py file, in if elif block, in lines: 106-111

Also, in general, if the number of H-clients varies (from 2), then modify the server.py accordingly.

# Dry Run for firs 8 rounds

## Round 1-4

These rounds will be used to setup the secure communication channels between the clients via the server. At the moment, these are empty (their implementation is pending)

## Round 5

SERVER sends model parameters to H-clients

SERVER sends dZ1 (received from the H-clients, in parts) to V-client (it will zeros in the first round)

H-CLIENTS receives model parameters

V-CLIENT receives dZ1

V-CLIENT sends Z1 to the server (in encrypted form)

H-CLIENTs do not send anythin to server

SERVER receives Z1 from V-client

## Round 6

SERVER sends part of Z1 (received from V-client) to H-clients

SERVER sends complete Z1 back to V-client (nothing to do further)

H-CLIENTS receives part of Z1

H-CLIENTS decrypts the part of Z1

Each H-CLIENT does local training

Each H-CLIENT sends local model parameters to the server, with masking

Each H-CLIENT sends dZ1 to the server (to send it to the V-client)

SERVER aggregates the model parameters

SERVER aggregates the dZ1

V-CLIENT recives Z1 to the server (do nothing)

## Round 7

SERVER sends updated aggregated model parameters to H-clients

SERVER sends dZ1 (received from the H-clients, in parts) to V-client (updated one)

H-CLIENTS receives model parameters

V-CLIENT receives dZ1

V-CLIENT updates its model parameters

V-CLIENT sends Z1 to the server, in encrypted form

H-CLIENTs do not send anythin to server

SERVER receives Z1 from V-client

## Round 8

SERVER sends part of Z1 (received from V-client) to H-clients

SERVER sends complete Z1 back to V-client (nothing to do further)

H-CLIENTS receives part of Z1

H-CLIENTS decrypts the part of Z1

Each H-CLIENT does local training

Each H-CLIENT sends local model parameters to the server, with masking

Each H-CLIENT sends dZ1 to the server (to send it to the V-client)

SERVER aggregates the model parameters

SERVER aggregates the dZ1

V-CLIENT recives Z1 to the server (do nothing)