import socket

# Set up the client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Server's host and port
host = '115.146.84.72'  # Assuming the server is running on the same machine
port = 5010

# Connect to the server
client.connect((host, port))

# Receive the welcome message from the server
welcome_message = client.recv(1024).decode()
print(f"Message from server: {welcome_message}")

# Send a message to the server
client.send("Hello, server!".encode())

# Close the connection
client.close()
