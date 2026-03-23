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

from shaun_clarke_csc6314 import load_and_prep_data

load_and_prep_data()