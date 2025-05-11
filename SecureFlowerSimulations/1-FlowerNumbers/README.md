# Flower Client-Server Setup for Federated Learning

## Purpose

This implementation demonstrates a simple federated learning setup using the Flower framework. The primary goal is to show how the server sends different matrices to clients based on their identity. Specifically:

- **V-client** receives a matrix that is **twice the average matrix**.
- **Other clients** (e.g., `client_H_1`, `client_H_2`) receive the **same average matrix**.

### Flow of Operations

1. **Server Customization**: The server creates a matrix for each client. `client_V` receives a customized matrix (twice the average of the previous matrices), while the other clients receive the same average matrix.
2. **Client Processing**: Each client receives its matrix, updates its local data, and sends results back to the server.
3. **Server Aggregation**: The server aggregates the results and prepares for the next round.

This setup illustrates how to customize the model for different clients within a federated learning framework.

For further implementation details, refer to the code's inline comments.
