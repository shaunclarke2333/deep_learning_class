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

# This can be onsidered the source of truth, for remapping.
# This is where it gets tricky. CrossEntropyLoss is expecting a sequence of numbers for the class labels(number of lables map to number of output neurons) that starts at 0.
# So if we use the actual indexes 2,3,5,7 from the CIFAR10 dataset, you quickly see how this becomes a problem.
LABEL_MAP: Dict[int, int] = {
    2: 0, # bird (CIFAR index 2)
    3: 1, # cat (CIFAR index 3)
    5: 2, # dog (CIFAR index 5) 
    7: 3, # horse (CIFAR index 7) 
}

# Remapping the label index to animal names, for decoding the predictions after training.
IDX_TO_CLASS: Dict[int, str] = {
    0: "bird", 1: "cat", 2: "dog", 3: "horse"
}

# Defining hyperparameters
BATCH_SIZE: int  = 64
NUM_EPOCHS: int  = 100
LEARNING_RATE: float = 1e-3
NUM_CLASSES: int = 4
# Where the trained model weights will be saved
MODEL_SAVE_PATH: str = "animal_cnn.pth"

# My machine has a GPU so i am allowing pytorch to automatically detect it.
# If there is no GPU, use the cpu.
# I learned real quick the model and the data must be on the same device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Defining the classifier head. this will map features to class score, aka which animal matches these featuers.
        self.classifier = nn.Sequential(
            # Randomly zero out 30% of the neurons only during training oof course.
            nn.Dropout (0.3),
            nn.Linear(128, num_classes)
        )


    # Defining the forward pass through the network
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        This method defines the forward pass through the CNN.
        I try to simplify it like this in my head:
        - self.features = what is in the image, create feature map, pool features 
        - torch.flatten = taked pooled featuers as input, flatten it into a 1D array
        - self.classifier = taked the 1D array of features, map the features to a class score(an animal matches the features)
        """

        # Running the CNN for feature extraction, whihc includes the Conv2d, BatchNorm2d, ReLU and AdaptiveAvgPool2d
        x = self.features(x)
        # Flattening the extracted features to change the shape from a 2D to a 1D vector.
        # So we can pass it into the classifier
        x = torch.flatten(x, 1)
        # Passing the flattened 1D featuers into the fully connected layer for a prediction.
        # But it output logits, whihc are raw scores and not probabilities.
        x = self.classifier(x)
        
        return x


#=============================================================
#4. Data Acquisition & Filtering 
#=============================================================

# This function modifies the dataset in place from the original CIFAR10 indices to 0,1,2,3
def remap_labels(dataset, label_map: Dict[int, int]) -> None:

  """
  """

  # Looping through dataset.targets to replace each lable using the label map
  for i in range(len(dataset.targets)):
    dataset.targets[i] = label_map.get(dataset.targets[i], dataset.targets[1])

# This function filters the dataset to only include the 4 animal classes we are working with.
def get_animal_subset(dataset, valid_original_indices: List[int]) -> Subset:


  """
  """
  # print(type(dataset.targets[0]), dataset.targets[0])
  # Using a list comprehension to capture the filtered results in a cleaner way.
  indices = [
      i for i,t in enumerate(dataset.targets)
      if t in {0, 1, 2, 3}
  ]

  return Subset(dataset, indices)


# The functions preps and returns train and test dataloaders for the 4 class animal subset
def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
  """
  """
  transform = transforms.Compose([
        # Resize all images to 64×64
        transforms.Resize((64, 64)),
        # Augment data, 50% chance to flip
        transforms.RandomHorizontalFlip(),
        # Convert the PIL image to a float tensor    
        transforms.ToTensor(),
        # Now we normalize using ImageNet channel stats                
        transforms.Normalize(
            # Per channel mean (R, G, B)                 
            (0.485, 0.456, 0.406),
            # Per channel std  (R, G, B)            
            (0.229, 0.224, 0.225)             
        )
    ])
  
  # Getting list of remapped indices
  valid_remapped: List[int] = list(LABEL_MAP.values())
  # Downloading the CIFAR-10 train dataset to my CWD
  train_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=True,
      download=True, transform=transform
  )
  # Downloading the CIFAR-10 test dataset to my CWD
  test_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=False,
      download=True, transform=transform
  )

  # Updating the labels in test and train for the 4 classes we are using in the CIFAR10 dataset
  # With our 0-3 indices
  remap_labels(train_dataset, LABEL_MAP)
  remap_labels(test_dataset, LABEL_MAP)

  # Getting animal seubsets for the train and test dataset
  train_dataset = get_animal_subset(train_dataset, valid_remapped)
  test_dataset = get_animal_subset(test_dataset, valid_remapped)

  # Packaging data in dataloaders
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  return train_loader, test_loader


# This is a training helper function
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
  """
  
  """
  model.train()
  total_loss: float = 0.0
  correct: int = 0
  total: int = 0

  # Switching the model to train mode, so it will utilize batchnorm and dropout
  model.train()
  # Creating the training loop
  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    # Removing gradients from the last batch
    optimizer.zero_grad()
    # The forward pass that will return the logits
    outputs = model(images)
    # Calculating the loss wit CrossEntropy to compare the output to the actual labels
    loss = criterion(outputs, labels)
    # Backpropagation to compute gradients
    loss.backward()
    # Updating the weights with the gradients computed by backprop
    optimizer.step()

    # Incrementing total loss
    total_loss += loss.item()
    # Making predictions, basically choosing the classes with the highest score for each image
    predictions = torch.argmax(outputs, dim=1)
    # Imcrementing correct predictions
    correct += (predictions == labels).sum().item()
    # incrementing the batch size
    total += labels.size(0)

  return total_loss / len(loader), 100.0 * correct / total


# This is an evaluation helper function
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch) -> Tuple[float, float]:
    """
    """
    # Switching the model to eval mode, so it not use, optimizer.zero_grad(), loss.backward(), optimizer.step()
    model.eval()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        
      # Creating the training loop
      for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # The forward pass that will return the logits
        outputs = model(images)
        # Calculating the loss wit CrossEntropy to compare the output to the actual labels
        loss = criterion(outputs, labels)
       
        # Incrementing total loss
        total_loss += loss.item()
        # Making predictions, basically choosing the classes with the highest score for each image
        predictions = torch.argmax(outputs, dim=1)
        # Imcrementing correct predictions
        correct += (predictions == labels).sum().item()
        # incrementing the batch size
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total



#=============================================================
#5. CWD Inference Engine
#=============================================================

def predict_single_image(model: nn.Module, image_path: str, device: torch.device) -> None:
    """
    """
    # Similar inference transform totraining except for RandomHorizontalFlip 
    infer_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Getting image and converting it to RGB to ensure a 3 channel input
    img = Image.open(image_path).convert("RGB")
    img_tensor = infer_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        class_name = IDX_TO_CLASS[pred_idx.item()]
        description = ANIMAL_INFO[class_name]

    print(f"# TARGET FILE: {os.path.basename(image_path)}")
    print(f"# PREDICTION:  {class_name.upper()}")
    print(f"# CONFIDENCE:  {confidence.item() * 100:.2f}%")
    print(f"# DESCRIPTION: {description}")


def main() -> None:
  """
  """


  # The data
  train_loader, test_loader = get_dataloaders()

  # The initializing the model and moving all model params to my GPU
  model = AnimalCNN(num_classes=NUM_CLASSES).to(DEVICE)

  # The loss
  criterion = nn.CrossEntropyLoss()

  # The optimizer
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

  # The training loop
  for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
  # Saving the model
  torch.save(model.state_dict(), MODEL_SAVE_PATH)
  print(f"Model saved to {MODEL_SAVE_PATH}")


  # Check if file exists
  target_file: str = "test_animal_image.jpg"
  image_path: str = os.path.join(os.getcwd(), target_file)

  if os.path.exists(image_path):
      print("\n" + "="*50)
      predict_single_image(model, image_path, DEVICE)
      
  else:
      print(f"\n[INFO] No '{target_file}' found in {os.getcwd()}")
      print("[INFO] Place a .jpg image in the current directory to test inference.")


if __name__=="__main__":
  main()

