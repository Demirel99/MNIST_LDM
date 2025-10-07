import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os # --- ADDED --- To create directories

# --- 1. Define Constants and Configuration ---
# These paths should match the directory structure you created
train_dir = 'data/train/images'
test_dir = 'data/test/images'

# Training parameters
img_size = 28
batch_size = 128
epochs = 10
learning_rate = 0.001

# --- ADDED: Define where to save the best model ---
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True) # Create the directory if it doesn't exist
save_path = os.path.join(SAVE_DIR, 'best_cnn_model.pth')

# --- 2. Setup Device (GPU or CPU) ---
# This makes the code portable and uses GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. Define Data Transformations and Load Data ---
# Define the sequence of transformations to be applied to the images.
# ToTensor() converts images to PyTorch Tensors and scales pixel values to [0, 1].
# Normalize() adjusts the tensor values to have a mean of 0.5 and a std of 0.5.
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Ensure images are single-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Mean and std for single channel
])

# Use ImageFolder to load data from the structured directories.
# It automatically finds classes from folder names.
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

# Create DataLoaders to handle batching, shuffling, and parallel data loading.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get class names
class_names = train_dataset.classes
print(f"Found classes: {class_names}")

# --- 4. Build the CNN Model ---
# In PyTorch, models are defined as classes inheriting from nn.Module.
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Classifier (fully connected layers)
        # After two max-pooling layers of size 2, a 28x28 image becomes 7x7.
        # So, the input features to the fc1 layer will be 64 * 7 * 7.
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Pass input through the first conv block
        x = self.pool1(self.relu1(self.conv1(x)))
        # Pass through the second conv block
        x = self.pool2(self.relu2(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7) # or torch.flatten(x, 1)
        # Pass through the classifier
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output raw scores (logits)
        return x

# Instantiate the model and move it to the selected device
model = CNN(num_classes=len(class_names)).to(device)

# --- 5. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss() # This loss function combines LogSoftmax and NLLLoss.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. Training and Validation Loop ---
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# --- ADDED: Variable to track best validation accuracy ---
best_accuracy = 0.0

print("\nStarting model training...")
for epoch in range(epochs):
    # --- Training Phase ---
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Use tqdm for a nice progress bar
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad(): # No need to calculate gradients during validation
        test_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
        for images, labels in test_progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    epoch_test_loss = running_loss / len(test_dataset)
    epoch_test_acc = correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)
    
    print(f"Epoch {epoch+1}/{epochs} -> "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")
    
    # --- ADDED: Check and save the best model ---
    if epoch_test_acc > best_accuracy:
        best_accuracy = epoch_test_acc
        # We save the model's 'state_dict' which contains all learnable parameters.
        torch.save(model.state_dict(), save_path)
        print(f"-> New best model saved to {save_path} with accuracy: {best_accuracy:.4f}")


print("Training finished.")

# --- 7. Plotting Results ---
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# # --- ADDED: 8. (Optional) How to Load the Best Model for Inference ---
# print("\nLoading the best model for evaluation...")

# # First, instantiate a new model of the same class
# # It's crucial that the architecture is identical to the one you saved.
# loaded_model = CNN(num_classes=len(class_names)).to(device)

# # Then, load the state dictionary from the saved file.
# loaded_model.load_state_dict(torch.load(save_path))

# # Set the model to evaluation mode. This is important for layers
# # like Dropout and BatchNorm, which behave differently during training and inference.
# loaded_model.eval()

# print("Model loaded successfully. You can now use 'loaded_model' for inference.")
# # For example, you can re-run validation on the test set to confirm performance:
# # with torch.no_grad():
# #     # ... (validation loop code here using loaded_model) ...