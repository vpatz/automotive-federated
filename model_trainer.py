import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader, TensorDataset

# Step-1: Generate random training data

NUM_SAMPLES = 10000
INPUT_DIM = 7 

# Define Input Parameters
# These are key inputs identified in research for accurate range prediction:
# 1. State of Charge (SOC) (%)
# 2. Battery Voltage (V)
# 3. Battery Temperature (C)
# 4. Current Vehicle Speed (km/h or mph)
# 5. Average Speed over the last 5 minutes
# 6. Current draw (A) (can be inferred or included directly)
# 7. State of Health (SOH) (%) (optional, useful for long-term accuracy)


# sample input data generated is normalized (zero mean, unit variance) 
EVdata = torch.randn(NUM_SAMPLES, INPUT_DIM) # 100 samples, 10 features each
#EVrange = torch.rand(0, 10, (NUM_SAMPLES, )) # 100 labels (0 or 1)
#EVrange = torch.randn(NUM_SAMPLES, INPUT_DIM) # 100 labels (0 or 1)

# sample output range in between (MIN_RANGE, MAX_RANGE)
MIN_RANGE = 0
MAX_RANGE = 100
EVrange = torch.FloatTensor(NUM_SAMPLES,1 ).uniform_(MIN_RANGE, MAX_RANGE) 

# Step-2: Create an instance of the Dataset

#dataset = CustomDataset(data, labels)
BATCH_SIZE = 100
dataset = TensorDataset(EVdata, EVrange)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Step-3: Define a multi-layer feed forward neural network

class EVRangePredictor(nn.Module):
    """
    A PyTorch model for predicting the remaining distance to empty (RDE) of an EV.
    """
    def __init__(self, input_dim):
        super(EVRangePredictor, self).__init__()
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

# --- Example Usage ---



# Initialize the model
model = EVRangePredictor(INPUT_DIM)
print("Model Architecture:")
print(model)

# 2. Prepare Sample Input Data (as a PyTorch Tensor)
# Data should be normalized or scaled before training in a real scenario
# Example single input: SOC=80%, Voltage=380V, Temp=25C, Speed=60km/h, AvgSpeed=55km/h, Current=50A, SOH=95%
sample_input = torch.tensor([80.0, 380.0, 25.0, 60.0, 55.0, 50.0, 95.0], dtype=torch.float32).unsqueeze(0) # unsqueeze(0) adds a batch dimension

# 3. Get a prediction (before training, this will be random)
with torch.no_grad():
    prediction = model(sample_input)
    print(f"\nSample Input: {sample_input.numpy().flatten()}")
    print(f"Predicted Remaining Distance (untrained): {prediction.item():.2f} km/miles")

# 4. Training Loop (Conceptual Outline)
# You would need a dataset of historical driving data (inputs and actual RDE/distance traveled)
# to train this model effectively.

# Define Loss function and Optimizer
criterion = nn.MSELoss() # Mean Squared Error for regression tasks
#criterion = nn.NLLLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Conceptual Training Process (requires actual data)
num_epochs = 100 
for epoch in range(num_epochs):
    for inputs, targets in dataloader: # Dataloader loads your historical data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
