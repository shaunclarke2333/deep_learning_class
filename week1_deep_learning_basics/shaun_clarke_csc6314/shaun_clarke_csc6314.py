"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: Project 1

In this project, you will implement and compare a traditional Machine Learning system (Logistic Regression)
against a Deep Learning system (Multi-Layer Perceptron) using PyTorch. Submit a single .py file.

"""
from typing import Tuple, List
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
from sklearn.metrics import roc_auc_score, accuracy_score
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


##### Requirement 2 #####
# This class implement's logistic regression as a neural network.
class LogisticRegressionML(nn.Module):
    """
    This class inherits from the nn.module class.
    This class implements logistic regression as a neural network
    Using nn.linear followed a sigmoid function that outputs pass through to produce probabilities.
    There are no hidden layers and this limits the model to only being able to learn a linear decision boundary.
    """

    def __init__(self, input_dim):
        """
        Using super to call the constructor of the parent class and initialize it
        so the LogisticRegressionML subclass can use it's methods.
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
        The shape of x coming into nn.linear is (N, D) meaning number of rows or samples, number of features or columns
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

##### Requirement 3 #####
# This class is the deep learning multi layer perceptron


class DeepMLP(nn.Module):
    """
    This class inherits from the nn.module class.
    This class implements a deep multi layer perceptron using nn.sequential which acts like a container
    that lets you stack layers in order(nn.linear, ReLU, Linear, Sigmoid) so that data can flow through them automatically
    which simplifies the forward method compared to the LinearRegressionModel.
    """

    def __init__(self, input_dim):
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
    def forward(self, x: torch.tensor) -> int:

        return self.net(x)


##### Requirement 4 #####
# This function counts the total trainable parameters in the model
def count_parameters(model) -> int:
    """
    The purpose of this function is to count the trainable
    parameters in the model.
    We will achieve this using list comprehension and summing the output:
    we will loop through model.parameters, all the learnable parameters in the model.
    Then access numel() to get the number of elements in each parameter tensor. While using
    requires_grad() as a filter so we only get parameters that were trainable, meaning they were updated during training.

    """

    # Returning the total trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# This function trains a model and returns its loss history
def train_model(model: nn.Module, X_train_tensor: torch.tensor, Y_train_tensor: torch.tensor, epochs: int = 20, lr: float = 0.002, batch_size: int = 64) -> Tuple[list[float], int]:
    """
    Purpose of this func is to traina model to save and return the loss history
    So we can analyze them in the later sections.
    The flow will look like this:
    - Run the training loop
    - keep track of the loss at each epoch()
    - return all the loss values for later analysis

    Args:
        model : nn.Module
        - the model to train (my LogisticRegressionML or DeepMLP model)

        X_train : FloatTensor
        - training features

        y_train : FloatTensor
        - binary labels, shape [N, 1]

        epochs : int
        - how many full passes over the training data

        lr : float
        - learning rate for Adam optimizer

        batch_size : int
        - number of samples per mini batch
 
    Returns:
        loss_history : list[float]
        - average loss per epoch (for plotting)
        train_time : float
        - total wall-clock training time in seconds
    """

    "The loss function"
    # Initializing the BCELoss function. We are using BCE because it is built for probabilities and it gives a stronger gradient
    criterion: nn.BCELoss = nn.BCELoss()

    "The optimizer"
    #Initializing the Adam Optimizer with model.Patameters(model parameters) and  lr=lr (specific learning rate)
    # We are using Adam and instead of SGD, because ADAM(adaptive + momentum) is able to adjust the learning rate (step size)
    # per parameter based on previous gradients. Unlike SGD that has no memory of previous gradients
    # and applies the same learning rate(step size) to all parameters. Adam keeps an ongoing estimate of the mean and variance of gradients
    # to make more efficient and stable updates.
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr)

    "The DataLoader"
    # Here we will be wrapping our training data(features and labels) in TensorDataset and DataLoader
    # The reason for the wrapping is pretty straight forward:
    # TensorDataset:
    # Combines inputs and labels TensorDataset(X_trainm Y_train) so each sample(row of data)
    # is a pair (x, y)
    # DataLoader:
    # The Dataloader receives the combined(x,y) dataset and handles feeding it to the model in batches, itterating and shufflin
    # of the dataset at the start of each epoch.
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    print(X_train_tensor.shape)
    # Preping the data to be loaded
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    "The training loop"
    # Empty list to keep track of th eloss histry, which is simply the loss values after each epoch.
    loss_history: List = []
    # Setting the model to training mode
    model.train()

    # Starting the clock before the training loop so we can measure how long training took
    start_train = time.time()

    # Starting the training loop
    for epoch in range(epochs):
        """
        We accumulate the loss for each batch into epoch_loss and count the number of batches using num_batches.
        So this way we can calculate the average loss for each epoch.
        """
        # Tracking losses across all batches in an epoch.
        epoch_loss = 0.0
        # Tracking number of batches to compute average loss per epoch
        num_batches = 0

        # Iterating over the DataLoader dataset train_dataloader to extract the inputs to train the model
        for X_batch, Y_batch in train_dataloader:
            "Training loop steps for my mental model"
            "step 1"
            # Calling optimizer.zero_grad to remov epreviously accumulated gradients
            optimizer.zero_grad()
            "step 2"
            # Running a forward pass of the batched features through the model
            outputs = model(X_batch)
            "step 3"
            # Calculating the loss(measuring how worng the model is) for this batch, by checking the model prediction against the target variable Y_batch
            loss = criterion(outputs, Y_batch)
            "step 4a"
            # Computing the gradient of the loss using backpropagation
            loss.backward()
            "step 4b"
            # Updating the weights based on the gradient of the loss that was calculated in step 4a
            # Basically adjusting the weights to reduce the error next time.
            optimizer.step()

            # accumulating loss across all the batches in this epoch to get total loss for the epoch.
            epoch_loss += loss.item()
            # Counting how many batches were processed.
            num_batches += 1
            
        # Calculating the average loss for each epoch. so at the end of this inner loop
        # Which is a full pass of all the data through the model(an epoch), the average loss of that epoch is calcualted.
        avg_loss = epoch_loss/num_batches
        # Keeping track of loss history by adding the avg_loss calculated in prev step to the loss_history list
        loss_history.append(avg_loss)
        # Printing the loss history in real time per epoch
        print(f"  Epoch {epoch+1:>3}/{epochs} | Loss: {avg_loss:.4f}")

    # Calculating how long training took
    train_time = time.time() - start_train

    return loss_history, train_time

# This function is the model evaluation engine
def evaluate_model(model, X_test_tensor, Y_test_tensor):
    """
    The purpose of this function is to evaluate the model using the test dataset.
    and return the accuracy, AUC-ROC, and inference latency.
    args:
        model: the trained nn.module
        X_test: the test feature torch.tensor dataset
        Y_test: the test target torch.tensor binary labels dataset. shape [N, 1]
    
    returns:
        accuracy: float
        - percentage of correct predictions
        acu: float
        - AUC-ROC score
        latency_ms: float
        - avg inference time per sample in milliseconds

    """

    # Switching the model to evaluation mode. 
    # This allows it disable training specific layers like dropout and batchnorm if we were using them
    model.eval()

    # torch.no_grad turns off gradient tracking because pytorch's autograd system automatically tracks the gradient.
    # Thhis is evaluation and not training so the autograd feature is not needed. This also saves a lot of memory and compute.
    with torch.no_grad():

        # Getting a thousand samples(rows of data) to use for a timed forward pass
        # This is saying, if the test set(X_test) is greater than or equal to 1000 use 1000 samples.
        # if the test set(X_test is smaller than 1000 use the full set)
        n_samples = min(1000, X_test_tensor.shape[0])

        # Starting the clock
        start_inf = time.time()
        # Doing a forward pass with the first 1000 samples from X_test.
        _ = model(X_test_tensor[:n_samples])

        # Calculating the avg time per sample in milliseconds.
        latency_ms = ((time.time() - start_inf) / len(X_test_tensor)) * 1000

        # Turning the model outputs into predictions and comparing them to the target variables
        # Getting the model output and conveting it to a numpy array.
        y_probs = model(X_test_tensor).numpy()
        # Applying a threshold of 0.5 to convert the model's probability output to a  binary(0,1) for class prediction
        y_pred = [1 if p >= 0.5 else 0 for p in y_probs]
        # Converting the test labels(Y_test_tensor) to a numpy array so they can be compared to the predictions
        y_true = Y_test_tensor.numpy()

        # calculating the accuracy score
        acc = accuracy_score(y_true, y_pred)

        # Calculating the roc_auc_score using probabilities and not predicitions.
        # Reason is, probabilities have a confidence score and they allow us to test multiple thresholds
        # Which is what the ROC curve is built from.
        auc = roc_auc_score(y_true, y_probs)

    return acc, auc, latency_ms


# This function generates a visualization with two subplots
def plot_results(ml_loss, dl_loss, ml_acc, dl_acc):
    """

    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # Loss curves plot
    ax1 = axes[0]
    ax1.plot(ml_loss, label='Logistic Regression (ML)')
    ax1.plot(dl_loss, label='Deep MLP (DL)')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy bar chart
    models = ['ML', 'DL']
    accuracies = [ml_acc, dl_acc]

    ax2 = axes[1]
    ax2.bar(models, accuracies)
    ax2.set_title('Model Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    
    # Using tight layout to automatically adjust spacing between the plots so everything fits nice and looks nice 
    plt.tight_layout()
    # Saving an image of the plot
    plt.savefig('project1_results.png', dpi=150)
   
    print("  Saved: project1_results.png")
    # Displaying the plot. it will automatically pause until the user closes the window
    plt.show()

# This function prints the formatted comaprison table
def print_results_table(ml_metrics, dl_metrics):
    print("\n" + "="*75)
    print("  COMPARISON RESULTS: Logistic Regression vs Deep MLP")
    print("="*75)

    # This makes all header row columns line up vertically regardless of value length
    print(f"{'Model':<25} {'Train Time (s)':>14} {'Latency (ms)':>14} {'Params':>10} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("-"*75)

    # Roundig the ML row floats to two decimal places with :.2f.
    # Then using :>14 to right-align the values to match the header column width
    print(f"{'Logistic Regression (ML)':<25} "
          f"{ml_metrics['train_time']:>14.2f} "
          f"{ml_metrics['latency_ms']:>14.4f} "
          f"{ml_metrics['params']:>10,} " # adds a comma to separate  thousands
          f"{ml_metrics['accuracy']:>10.4f} "
          f"{ml_metrics['auc']:>10.4f}")

    # Using the same format for this DL row, just different data
    print(f"{'Deep MLP (DL)':<25} "
          f"{dl_metrics['train_time']:>14.2f} "
          f"{dl_metrics['latency_ms']:>14.4f} "
          f"{dl_metrics['params']:>10,} "
          f"{dl_metrics['accuracy']:>10.4f} "
          f"{dl_metrics['auc']:>10.4f}")

    print("="*75)



def main():
    """
    This runs the full pipeline:
    1. Load and prepare the data
    2. Build both models
    3. Train both models
    4. Evaluate both models
    5. Print the results table
    6. Plot theh results
 
    """
 
    print("="*50)
    print(" ML vs DL: California Housing Binary Classification")
    print("="*50)
 
    # Loading and preparing data
    print("\nLoading and preparing the data...")
    X_train, X_test, y_train, y_test, input_dim = load_and_prep_data()
    print(f"  input_dim={input_dim}, train size={X_train.shape[0]}, test size={X_test.shape[0]}")
 
    # Building the models
    print("\nBuilding models...")
    ml_model = LogisticRegressionML(input_dim)
    dl_model = DeepMLP(input_dim)
 
    ml_params = count_parameters(ml_model)
    dl_params = count_parameters(dl_model)
    print(f"  ML params: {ml_params:,}  |  DL params: {dl_params:,}")
 
    # Training the Logistic Regression (ML)
    print("\nTraining Logistic Regression (ML)...")
    ml_loss, ml_train_time = train_model(
        ml_model, X_train, y_train, epochs=20, lr=0.002, batch_size=64
    )
    
    # Training the Deep MLP (DL)
    print("\n       Training Deep MLP (DL)...")
    dl_loss, dl_train_time = train_model(
        dl_model, X_train, y_train, epochs=20, lr=0.002, batch_size=128
    )
 
    # Evaluating the models
    print("\nEvaluating models...")
    ml_acc, ml_auc, ml_latency = evaluate_model(ml_model, X_test, y_test)
    dl_acc, dl_auc, dl_latency = evaluate_model(dl_model, X_test, y_test)
 
    print("\nResults:")
    # ML results
    ml_metrics = {
        "train_time": ml_train_time,
        "latency_ms": ml_latency,
        "params": ml_params,
        "accuracy": ml_acc,
        "auc": ml_auc,
    }

    # DL results
    dl_metrics = {
        "train_time": dl_train_time,
        "latency_ms": dl_latency,
        "params": dl_params,
        "accuracy": dl_acc,
        "auc": dl_auc,
    }
    # Displaying model results
    print_results_table(ml_metrics, dl_metrics)
    # Plotting results and saving image.
    plot_results(ml_loss, dl_loss, ml_acc, dl_acc)
 
 

if __name__ == "__main__":
    main()















    
    