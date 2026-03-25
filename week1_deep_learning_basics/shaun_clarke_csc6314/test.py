# The pytorch tensor core engine
import torch
# The neural network module. This is the base class all models will inherit from
import torch.nn as nn
# These are the optimization algorithms that update the model's parameters during training. The engine that learns
import torch.optim as optim
# The data pipeline tools: TensorDataset pairs (x) features and (y) labels together.
# DataLoader: Splits the data into mini batches and iterates over them during training.
# Batches make training faster and is less memory intensive
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_california_housing # The dataset i will be working with
from sklearn.model_selection import train_test_split # Split the data into training and testing
from sklearn.preprocessing import StandardScaler # For standardizing ML features so they are all on the same scale mean=0 std=1
from sklearn.metrics import roc_auc_score # This computes the model's ability to discriminate between classes
import matplotlib.pyplot as plt

from shaun_clarke_csc6314 import load_and_prep_data, LogisticRegressionML, DeepMLP, train_model, evaluate_model

# Testing load and prep data
X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, input_dim = load_and_prep_data()

# Displaying values for X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, input_dim
print("=" * 50)
print(f"Displaying values for X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, input_dim")
print(f"X_train_tensor: {X_train_tensor}")
print(f"X_test_tensor: {X_test_tensor}")
print(f"Y_train_tensor: {Y_train_tensor}")
print(f"Y_test_tensor: {Y_test_tensor}")
print(f"input_dim: {input_dim}")
print("=" * 50)

# Testing the model after loading data
print("=" * 50)
print(f"Testing the model:")
print(f"Expected Output")
print(f"X_train_tensor.shape like (N, D)")
print(f"X_train_tensor.shape like (N, 1)")
model = DeepMLP(input_dim)
print(model)
print(X_train_tensor.shape)
print(Y_train_tensor.shape)
print("=" * 50)

# Testin one forward pass sanity check
print("=" * 50)
print(f"Testin one forward pass:")
print(f"Expected Output")
print(f"output shape should be (5, 1)")
print(f"output values should be between 0 and 1")
sample_output = model(X_train_tensor[:5])
print(f"Output shape: {sample_output.shape}")
print(f"Output values: {sample_output}")
print("=" * 50)

# training the model
train_model(model, X_train_tensor, Y_train_tensor)

# print(train_model(model, X_train_tensor, Y_train_tensor))
print(evaluate_model(model, X_test_tensor, Y_test_tensor))


