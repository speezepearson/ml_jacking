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

# Weights and biases
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print("weights and biases")
print(w)
print(b)

# Define the model
def model(x):
    return x * w.t() + b

# Generate predictions
preds = model(inputs)
print("predictions")
print(preds)
# Compare with targets
print("targets")
print(targets)

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Compute loss
loss = mse(preds, targets)
print("loss")
print(loss)

# Compute gradients
loss.backward()

# Gradients for weights
print("gradiants for weights")
print(w)
print(w.grad)

# Gradients for bias
print("gradiants for bias")
print(b)
print(b.grad)

print("reset to zero")
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

# Generate predictions
print("\ngenerate predictions again")
preds = model(inputs)
print(preds)

# Calculate the loss
print("\ncalc loss again")
loss = mse(preds, targets)
print(loss)

# Compute gradients
loss.backward()

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

# Calculate loss
print("calc loss after adjustment:")
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


print("\niterate a bunch of times")
# Train for 100 epochs
for i in range(10000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-2
        b -= b.grad * 1e-2
        w.grad.zero_()
        b.grad.zero_()

# Calculate loss
print("loss after iteration")
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

print("weights and biases")
print(w)
print(b)

print("\n preds and targets")
print(preds)

print(targets)


