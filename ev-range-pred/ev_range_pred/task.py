"""ev-range-pred: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


class Net(nn.Module):
    """EV Range Prediction Model - Multi-layer feed forward neural network"""
    
    def __init__(self, input_dim=7):
        super(Net, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.relu1 = nn.ReLU()                # ReLU activation function
        self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
        self.relu2 = nn.ReLU()                # ReLU activation function
        self.fc3 = nn.Linear(64, 1)           # Output layer (1 neuron for RDE prediction)

    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Global constants for dataset generation
NUM_SAMPLES = 10000
INPUT_DIM = 7
MIN_RANGE = 0
MAX_RANGE = 100
BATCH_SIZE = 32

# Cache for generated dataset
_cached_dataset = None


def generate_ev_dataset(num_samples=NUM_SAMPLES, input_dim=INPUT_DIM):
    """
    Generate synthetic EV range prediction dataset.
    
    Input features (7 dimensions):
    1. State of Charge (SOC) (%)
    2. Battery Voltage (V)
    3. Battery Temperature (C)
    4. Current Vehicle Speed (km/h or mph)
    5. Average Speed over the last 5 minutes
    6. Current draw (A)
    7. State of Health (SOH) (%)
    
    Output: Remaining distance/range (km/miles)
    """
    # Generate normalized input data (zero mean, unit variance)
    EVdata = torch.randn(num_samples, input_dim)
    
    # Generate output range between MIN_RANGE and MAX_RANGE
    EVrange = torch.FloatTensor(num_samples, 1).uniform_(MIN_RANGE, MAX_RANGE)
    
    return EVdata, EVrange


def load_data(partition_id: int, num_partitions: int):
    """Load partitioned EV range prediction data."""
    global _cached_dataset
    
    # Generate or retrieve cached dataset
    if _cached_dataset is None:
        EVdata, EVrange = generate_ev_dataset()
        _cached_dataset = TensorDataset(EVdata, EVrange)
    
    # Calculate partition size
    total_samples = len(_cached_dataset)
    samples_per_partition = total_samples // num_partitions
    
    # Create partition indices
    start_idx = partition_id * samples_per_partition
    if partition_id == num_partitions - 1:
        # Last partition gets remaining samples
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_partition
    
    # Get partition subset
    partition_indices = list(range(start_idx, end_idx))
    partition_data = torch.utils.data.Subset(_cached_dataset, partition_indices)
    
    # Split partition into train (80%) and test (20%)
    partition_size = len(partition_data)
    train_size = int(0.8 * partition_size)
    test_size = partition_size - train_size
    
    train_data, test_data = random_split(
        partition_data, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.MSELoss().to(device)  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1
    avg_trainloss = running_loss / total_batches
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.MSELoss()
    net.eval()
    total_loss = 0.0
    total_mae = 0.0  # Mean Absolute Error
    total_samples = 0
    
    with torch.no_grad():
        for batch in testloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            
            # Calculate MSE loss
            loss = criterion(outputs, targets).item()
            total_loss += loss * len(inputs)
            
            # Calculate MAE for interpretability
            mae = torch.abs(outputs - targets).mean().item()
            total_mae += mae * len(inputs)
            
            total_samples += len(inputs)
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    # Return loss and MAE (as a proxy for "accuracy" metric)
    return avg_loss, avg_mae
