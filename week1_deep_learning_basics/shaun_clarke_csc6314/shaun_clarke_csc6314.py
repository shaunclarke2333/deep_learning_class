"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: Project 1

In this project, you will implement and compare a traditional Machine Learning system (Logistic Regression)
against a Deep Learning system (Multi-Layer Perceptron) using PyTorch. Submit a single .py file.

"""
from typing import Tuple
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
import sklearn
# The dataset i will be working with
from sklearn.datasets import fetch_california_housing
# Split the data into training and testing
from sklearn.model_selection import train_test_split
# For standardizing ML features so they are all on the same scale mean=0 std=1
from sklearn.preprocessing import StandardScaler
# This computes the model's ability to discriminate between classes
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


##### Requirement 1 #####
# This function loads and preps the data
def load_and_prep_data() -> Tuple:
    """
    """

    "Step1 : Loading the data"
    # Loading the housing dataset
    data: sklearn.utils._bunch.Bunch = fetch_california_housing()
    # Defining the features/inputs before preprocessing, hence the raw
    x_raw: np.ndarray = data.data
    # Defining the target variables that we are trying to predict. not yet converted to binary
    y_raw = data.target

    "Step 2: creating binary labels"
    # Creating the binary labels by converting all values in Y_raw above the 70th percentile ar represented as false
    # and teh rest labeld as true. then convert the boolean to binary 1,0.
    threshold: np.ndarray = np.percentile(y_raw, 70)
    y_binary: np.ndarray = (y_raw > threshold).astype(
        int)  # 1 = high value, 0 = not high value

    "Step 3: Train/Test split"
    # Splitting x_raw and y_binary into train test splits, using stratify to make sure the class distribution is preserved for bothe the training and test sets.
    X_train, X_test, Y_train, Y_test = train_test_split(
        x_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    "Step 4: Instantiate StandardScaler"
    # initializin StandardScaler
    scaler: StandardScaler = StandardScaler()
    # fitting and transforming the training data with standarscaler
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    # We will only perform a transform on the test data, to avoid data leakage(exposing test data to the model)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    "Step 5 convert np.ndarrays to pytorch tensors"
    # Converting the scaled(standardscaler) numpt arrays to pytorch float tensors.
    # Using".float" becasuse standardscalers output is float64 and nn.BCE and nn.Linear will throw an error unless it gets float32.(learned this the hardway and added this comment line after)
    # Floats because nn.BCELoss expects floats because it will be working with probabilities.
    # Uisng unsqueeze to add a dimension so the target variable data match the shape the model is expecting.
    # lets say the original shape of Y_train is (100,). The shape of the Pytorch model prediction output is like (100, 1).
    # So after unsqueeze we would have (100, 1)
    # So i see hwo quickly this can be a problem and cause errors or maybe not training the model correctly.
    X_train_tensor: torch.float32 = torch.tensor(X_train_scaled).float()
    X_test_tensor: torch.float32 = torch.tensor(X_test_scaled)
    Y_train_tensor: torch.float32 = torch.tensor(Y_train).float().unsqueeze(1)
    Y_test_tensor: torch.float32 = torch.tensor(Y_test).float().unsqueeze(1)

    # Getting the number of columns for the input dimensions for pytorch from the tuple returned by .shape (num_rows, num_cols) num_columns).
    # each column\feature is an input and each row is a sample.
    # The imput_dim tells pytorch how many features to expect for each sample.
    # Basically means, input_dim tells torch the number of input it's getting per row, which determines the size of the input layer.
    # Then the first layer creates the weights based on the size of the input layer.
    input_dim: int = X_train_tensor.shape[1]

    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor, input_dim

    # print(data.target)
    # print(X_train_tensor.shape)
    # print(type(input_dim))


# This class inherots from the nn.module class.
# This will allow us to implement logistic regression as a neural network.