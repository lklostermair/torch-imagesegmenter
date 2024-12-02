from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import accuracy_score, confusion_matrix
from torchnn import ImageClassifier
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load losses
with open("output/losses.pkl", "rb") as f:
    losses = pickle.load(f)

train_losses = losses["train_losses"]
val_losses = losses["val_losses"]

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))

# Define dataset transforms
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

# Load test data (only required for visualization)
test_data = torch.load(os.path.join(output_dir, "test_data.pt"))  # Save this in training script
test_loader = torch.load(os.path.join(output_dir, "test_loader.pt"))  # Save this in training script

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = ImageClassifier().to(device)
clf.load_state_dict(load(os.path.join(output_dir, 'best_model.pt'), map_location=device))
clf.eval()

# Predictions and labels
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        yhat = clf(X)
        preds = torch.argmax(yhat, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)

# Get correct and incorrect indices
correct_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred == true]
incorrect_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred != true]

# Visualization helper function
def visualize_samples(indices, title, save_path, correct=True):
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)
    for ax, idx in zip(axs.flatten(), indices[:4]):  # Show first 4 samples
        img, label = test_data[idx]
        img_np = np.transpose(img.numpy(), (1, 2, 0))  # HWC format
        pred_label = all_preds[idx]
        ax.imshow(img_np)
        
        # Get class names for true and predicted labels
        true_class = class_names[label]
        pred_class = class_names[pred_label]
        
        ax.set_title(f"True: {true_class}, Pred: {pred_class}", 
                     fontsize=12, 
                     color="green" if correct else "red")
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)


# Visualize correct samples
visualize_samples(
    correct_indices,
    f"Correctly Classified Test Samples (Accuracy: {accuracy * 100:.2f}%)",
    os.path.join(output_dir, "correctly_classified.png"),
    correct=True
)

# Visualize incorrect samples
visualize_samples(
    incorrect_indices,
    f"Wrongly Classified Test Samples (Accuracy: {accuracy * 100:.2f}%)",
    os.path.join(output_dir, "wrongly_classified.png"),
    correct=False
)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation="nearest", cmap="viridis")
plt.title("Confusion Matrix", fontsize=16)
plt.colorbar(label="Number of Samples", shrink=0.8)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
num_classes = cm.shape[0]
plt.xticks(np.arange(num_classes), labels=[str(i) for i in range(num_classes)], fontsize=12, rotation=45)
plt.yticks(np.arange(num_classes), labels=[str(i) for i in range(num_classes)], fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
