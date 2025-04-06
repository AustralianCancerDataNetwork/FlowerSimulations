
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score



def load_data():

    # Load data from CSV
    data = pd.read_csv("data/data_3.csv")

    # Shuffle the data
    data = shuffle(data)

    # Split data into input (X) and output (y)
    X = data.iloc[:, :4].values  # Input features
    y = data.iloc[:, 4].values   # Output feature

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

def train(model, train_data, epochs=10):
    learning_rate = 0.001
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
            
        epoch_loss = total_loss / len(train_data.dataset)  # Compute average loss per sample


from sklearn.metrics import accuracy_score

import torch.nn.functional as F

def test(model, test_data):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
            
            predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
            batch_accuracy = accuracy_score(labels.numpy(), predicted.detach().numpy())
            total_accuracy += batch_accuracy * labels.size(0)  # Multiply accuracy by batch size
            
            total_samples += labels.size(0)
    
    overall_loss = total_loss / total_samples
    overall_accuracy = total_accuracy / total_samples
    
    print(f'Loss on test set: {overall_loss:.4f}, Accuracy on test set: {overall_accuracy*100:.2f}%')
    
    return overall_loss, overall_accuracy  # Return both loss and accuracy





# Define the neural network class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # Fully connected layer 1
        self.relu = nn.ReLU()                           # Activation function
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)# Fully connected layer 2
        self.fc3 = nn.Linear(hidden_size2, output_size)# Fully connected layer 3

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
# Assuming you have X_train, y_train, X_test, and y_test as tensors
X_train, y_train, X_test, y_test = load_data()

# Create a TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create a DataLoader
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define model parameters
input_size = 4
hidden_size1 = 5
hidden_size2 = 5
output_size = 1

# Create the model
net = Net(input_size, hidden_size1, hidden_size2, output_size)


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="115.146.84.72:5010",
        client=FlowerClient().to_client(),
    )
