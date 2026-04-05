"""
Name: Shaun Clarke
Course: CSC6314 Deep Learning
Instructor: Margaret Mulhall
Assignment: Build and train a Convolutional Neural Network (CNN) to identify four specific animal species and deploy it as a functional script
that processes an image (test_animal_image.jpg) from your current working directory (CWD).
"""

from typing import Tuple, List, Dict
# The actual tensor library for pytorch that will be doing all the math.
import torch
# Where the actual models live                              
import torch.nn as nn
# Gradient descent optimizers like (Adam, SGD)                     
import torch.optim as optim
# This makes the CIFAR-10 dataset and the pretrained model zoo available.
import torchvision
# The image preprocessing pipeline that alloows us to resize and normalize
import torchvision.transforms as transforms  
# Python Imaging Library to open and read image files
from PIL import Image                     
import os
# The DataLoader alows us to feed the model data in batches is also subset filters it                              
from torch.utils.data import DataLoader, Subset  

#=============================================================
#2. Species Knowledge Base
#=============================================================

# Create a dictionary to map predicted class labels to human-readable descriptions
# These dictionaries together are like our decoder ring.
ANIMAL_INFO: Dict[str, str] = {
    "cat":   "Small carnivorous mammal with soft fur and retractable claws.",
    "dog":   "Domesticated carnivorous mammal known for loyalty.",
    "horse": "Large herbivorous mammal used for riding and racing.",
    "bird":  "Small animal with wings",
}


# Dictonary with the original CIFAR-10 dataset animal inexes and huma readable labels.
# This dict makes everyone happy, the model cant work with text and names make more sense for us humans.
# So this is why we have both index and names mapped here.
CIFAR10_ANIMAL_INDICES: Dict[str, int] = {
    "bird": 2, "cat": 3, "dog": 5, "horse": 7
}

# Remapping the label index to animal names, for decoding the predictions after training.
# This is where it gets tricky. CrossEntropyLoss is expecting a sequence of numbers for the class labels(number of lables map to number of output neurons) that starts at 0.
# So if we use the actual indexes 2,3,5,7 from the CIFAR10 dataset, you quickly see how this becomes a problem.
IDX_TO_CLASS: Dict[int, str] = {
    0: "bird", 1: "cat", 2: "dog", 3: "horse"
}

# Defining hyperparameters
BATCH_SIZE: int  = 64
NUM_EPOCHS: int  = 10
LEARNING_RATE: float = 1e-3
NUM_CLASSES: int = 4
# Where the trained model weights will be saved
MODEL_SAVE_PATH: str = "animal_cnn.pth"

# My machine has a GPU so i am allowing pytorch to automatically detect it.
# If there is no GPU, use the cpu.
# I figured learned real quick the model and the data must be on the same device.
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=============================================================
#3. Build the CNN Architecture
#=============================================================

# This class will be the container for the 4 class animal classifier
class AnimalCNN(nn.Module):
    """
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super(AnimalCNN, self).__init__()
        self.features = nn.Sequential(
            # 3 is input channels(rgb), 32 is feature maps, kernel_sze 3x3 filter, padding of 1 to preserve spatial size.Basically keeps the image soze the same
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Layer 2
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Layer 3
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Using AdaptiveAvgPool2d which is like saying how much of this feature exists in the image overall
            nn.AdaptiveAvgPool2d((1,1))
        )

        # Defining the classifier head
        self.classifier = nn.sequential(
            # Randomly zero out 30% of the neurons only during training oof course.
            nn.Dropout (0.3),
            nn.Linear(128, num_classes)
        )


        # Defining the forward pass through the network
        def forward(self, x: torch.tensor) -> torch.tensor:
            """
            This method defines the forward pass through the CNN
            """

            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

            return x








#=============================================================
#4. Data Acquisition & Filtering 
#=============================================================


#=============================================================
#5. CWD Inference Engine
#=============================================================

