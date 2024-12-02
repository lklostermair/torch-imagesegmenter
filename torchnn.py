from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pickle

output_dir = "output"  # Define the output directory
os.makedirs(output_dir, exist_ok=True)

# Define transformations (normalize CIFAR-10 images)
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    RandomCrop(32, padding=4),   # Randomly crop with padding
    ToTensor(),  # Convert images to PyTorch tensors
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Image Classifier Class
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout with p=0.5
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout with p=0.5
            nn.Linear(512, 10)
            
        )
    def forward(self, x):
        return self.model(x)
    
# Instance of the NN, loss, optimizer
# Training and saving logic (only when this file is run directly)
if __name__ == "__main__":
    # Instance of the NN, loss, optimizer
    clf = ImageClassifier().to('cpu')
    opt = Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    no_improve_count = 0 

    for epoch in range(15):
    # Training Loop
        train_loss = 0
        clf.train()  # Set model to training mode
        for X, y in train_loader:
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        # Average training loss for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation Loop
        val_loss = 0
        val_correct = 0  # To count correct predictions
        clf.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to('cpu'), y.to('cpu')
                yhat = clf(X)
                loss = loss_fn(yhat, y)
                val_loss += loss.item()
                
                # Count correct predictions
                preds = torch.argmax(yhat, dim=1)
                val_correct += (preds == y).sum().item()

        # Average validation loss for the epoch
        val_loss /= len(test_loader)
        val_accuracy = val_correct / len(test_data)  # Compute validation accuracy

        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(clf.state_dict(), "output/best_model.pt")
            print(f"Saved best model at epoch {epoch + 1}")
            no_improve_count = 0  # Reset the counter
        else:
            no_improve_count += 1  # Increment the counter

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy * 100:.2f}%")


    # Save the losses
    with open(os.path.join(output_dir, "losses.pkl"), "wb") as f:
        pickle.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Save the test dataset and DataLoader for visualization purposes
    torch.save(test_data, os.path.join(output_dir, "test_data.pt"))
    torch.save(test_loader, os.path.join(output_dir, "test_loader.pt"))

    print("Test dataset and DataLoader saved for visualization.")