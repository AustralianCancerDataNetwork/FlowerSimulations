There are eight folders for eight simulations using FLOWER, starting from the exchange of simple matrices to more advanced combined federated learning algorithms. 

Here is the overview (the details are in their respective folders):

1-FlowerNumbers: Demonstrates a basic Flower federated learning setup where the server sends customized matrices to clients based on identity, e.g., client_V receives twice the average matrix, while others get the average. Useful for understanding client-specific customization.

2-FlowerNumbers_alt_rounds: Extends the basic Flower setup by sending customized matrices to clients in alternating rounds for dynamic server-client interactions over time.

3-FlowerVFL_without_Encr: Implements vertical federated learning using Flower without encryption. Clients hold different data modalities, and the server mediates all communication in a multi-round training protocol.

4-FlowerCFL_VOutFed_without_Encr: Demonstrates a combined federated learning (horizontal + vertical) setup using Flower, where clients hold disjoint subsets of data samples and features. The server coordinates training across all clients without encryption. The output feature is with the V-client and not with the H-clients.

5-FlowerCFL_HOutFed_without_Encr: Implements the combined federated learning, however, the putput feature is with the H-clients instead of V-client.

6-FlowerVFL_with_Encr: Implements vertical federated learning with encryption and masking in the Flower framework (rest same as in 3-FlowerVFL_without_Encr).

7-FlowerCFL_VOutFed_with_Encr: Implements combined federated learning with encryption and masking in the Flower framework (rest same as in 4-FlowerCFL_VOutFed_without_Encr).

8-FlowerCFL_HOutFed_with_Encr: Implements combined federated learning with encryption and masking in the Flower framework (rest same as in 5-FlowerCFL_HOutFed_without_Encr).