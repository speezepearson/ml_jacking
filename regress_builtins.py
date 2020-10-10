import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


inputs = np.array([1,2,3,4,5,6,7,8,9,10], dtype='float32')
targets = np.array([1,1.111374373,1.119168756,8.898077878,6.425522226,23.80948737,25.21115499,2.421243748,10.8532073,4.14585736], dtype='float32')

print(inputs)
print(targets)

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)

# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader

# Define dataset
train_ds = TensorDataset(inputs, targets)

# Define data loader
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
#print(next(iter(train_dl)))

# Define model
model = nn.Linear(1, 1)
print("weight and bias")
print(model.weight)
print(model.bias)

# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

# Define loss function
loss_fn = F.mse_loss

# reshape stuff
inputs = torch.reshape(inputs, (10,1)) #why do we need to do this?
targets = torch.reshape(targets, (10,1)) 

"""
loss = loss_fn(model(inputs), targets)
print("\nLoss is:")
print(loss)
"""

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))
    print("weight and bias")
    print(model.weight)
    print(model.bias)

# Train the model 
fit(1000, model, loss_fn, opt)

# Generate predictions
preds = model(inputs)
print("predictions:")
print(preds)

