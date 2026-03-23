"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: Project 1

In this project, you will implement and compare a traditional Machine Learning system (Logistic Regression)
against a Deep Learning system (Multi-Layer Perceptron) using PyTorch. Submit a single .py file.

"""

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
from sklearn.datasets import fetch_california_housing # The dataset i will be working with
from sklearn.model_selection import train_test_split # Split the data into training and testing
from sklearn.preprocessing import StandardScaler # For standardizing ML features so they are all on the same scale mean=0 std=1
from sklearn.metrics import roc_auc_score # This computes the model's ability to discriminate between classes
import matplotlib.pyplot as plt



##### Requirement 1 #####
# This function loads and preps the data 
def load_and_prep_data():

    """
    Note to self: list steps to create mental model
    """
    "Step1 : Loading the data"
    # Loading the housing dataset
    data: sklearn.utils._bunch.Bunch = fetch_california_housing()
    # Defining the features/inputs before preprocessing, hence the raw
    x_raw: np.ndarray = data.data
    # Defining the target variables that we are trying to predict. not yet converted to binary
    y_raw = data.target

    # Creating the binary labels by converting all values in Y_raw above the 70th percentile ar represented as false
    # and teh rest labeld as true. then convert the boolean to binary 1,0.
    threshold: np.ndarray = np.percentile(y_raw, 70)  
    y_binary: np.ndarray  = (y_raw > threshold).astype(int)  # 1 = high value, 0 = not high value 

    # Splitting x_raw and y_binary into train test splits, using stratify to make sure the class distribution is preserved for bothe the training and test sets.
    X_train, X_test, Y_train, Y_test = train_test_split(x_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    # initializin StandardScaler
    scaler: StandardScaler = StandardScaler()
    # fitting and transforming the training data with standarscaler
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    # We will only perform a transform on the test data, to avoid data leakage(exposing test data to the model)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    "Step 5 convert np.ndarrays to pytorch tensors"
    

    # print(data.target)
    print(threshold)



