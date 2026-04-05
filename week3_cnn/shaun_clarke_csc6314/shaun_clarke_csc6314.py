"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: Build and train a Convolutional Neural Network (CNN) to identify four specific animal species and deploy it as a functional script
that processes an image (test_animal_image.jpg) from your current working directory (CWD).
"""

from typing import Tuple, List, Dict
# The core PyTorch tensor operations
import torch                          
import torch.nn as nn                 # Building blocks for neural networks (layers, loss)
import torch.optim as optim           # Optimizers: SGD, Adam, etc.
from torch.utils.data import DataLoader, Subset  # Batching + filtering datasets
import torchvision                    # Datasets and pretrained model zoo
import torchvision.transforms as transforms  # Image preprocessing pipeline
import torchvision.models as models   # Pretrained architectures (ResNet, VGG, etc.)
import os

