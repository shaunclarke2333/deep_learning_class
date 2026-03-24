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
import time
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
    X_test_tensor: torch.float32 = torch.tensor(X_test_scaled).float()
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



# This class implement's logistic regression as a neural network.
class LinearRegressionML(nn.Module):
    """
    This class inherits from the nn.module class.
    This class implements logistic regression as a neural network
    Using nn.linear followed a sigmoid function that outputs pass through to produce probabilities.
    There are no hidden layers and this limits the model to only being able to learn a linear decision boundary.
    """

    def __init__ (self, input_dim):
        """
        Using super to call the constructor of the parent class and initialize it
        so the LinearRegressionML subclass can use it's methods.
        """ 
        super().__init__()

        # Instantiating the linear layer with nn.linear with the number of dimensions input and number of outputs
        self.layer = nn.Linear(input_dim, 1)
        # Instantiating the sigmoid activation function that wil be use to convert linear layer outputs to a probability between 0 and 1.
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    # Defining the forward pass method that will define how data will move through the model and return an output.
    def forward(self, x: torch.tensor) -> Tuple:
        """
        This method simulates the data flowing through the network.
        The shape of x coming into nn.linear is (N, D) meaning number of rows\samples, number of features\columns
        nn.linear output shape is (N, 1) 1 value(logit) per row (N)
        That output is then passed into the sigmoid function which squashes the values between 0 and 1
        but the shape remmains the same (N, 1)

        returns (N, 1)
        """

        # Passing the input dimensions to the linear layer
        out = self.layer(x)
        # Passing the linear layer output into the sigmoid activation function
        out = self.sigmoid(out)

        return out
        

# This class is the deep learning multi layer perceptron
class DeepMLP(nn.Module):
    """
    This class inherits from the nn.module class.
    This class implements a deep multi layer perceptron using nn.sequential which acts like a container
    that lets you stack layers in order(nn.linear, ReLU, Linear, Sigmoid) so that data can flow through them automatically
    which simplifies the forward method compared to the LinearRegressionModel.
    """

    def __init__ (self, input_dim):
        """
        Using super to call the constructor of the parent class and initialize it
        so the DeepMLP subclass can use it's methods.
        """ 
        super().__init__()
        """
        Initializing nn.Sequential to stack the MLP layers automatically, with varying widths 128 -> 64 -> 32.
        This decreaing layer sizes forms a funnel shape architecture. This architecture allows the model to start wide
        capture lots of feature combinations and gradually reduce by focusing on important ptterns as it moves through the layers.
        This allows the model to learn hierarchical feature representation in an efficient way. 

        """
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass method that will pass our training data into self.net.
    def forward(self, x: torch.tensor):
        
        return self.net(x)






        


