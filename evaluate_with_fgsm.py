import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Configuration and Constants ---
# Ensure these paths and parameters match your training script
test_dir = 'data/test/images'
model_path = os.path.join('saved_models', 'best_cnn_model.pth')
img_size = 28
batch_size = 64 # You can adjust this for evaluation

# --- 2. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. Re-define the Model Architecture ---
# This is crucial. The class definition must be EXACTLY the same
# as the one used for training to load the state_dict correctly.
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 4. Load Data ---
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
class_names = test_dataset.classes
print(f"Found classes: {class_names}")

# --- 5. Load the Trained Model ---
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run the training script first to save the model.")
    exit()

model = CNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Set the model to evaluation mode
print(f"Model loaded from {model_path}")

# --- 6. Define the FGSM Attack Function ---
def fgsm_attack(model, loss_fn, images, labels, epsilon):
    """
    Performs the FGSM attack to generate adversarial examples.
    """
    # Set requires_grad attribute of tensor. Important for attack.
    images.requires_grad = True

    # Forward pass
    outputs = model(images)
    
    # Calculate the loss
    model.zero_grad()
    loss = loss_fn(outputs, labels)
    
    # Backward pass to get gradients of loss w.r.t inputs
    loss.backward()

    # Collect the gradient of the input data
    data_grad = images.grad.data
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + epsilon * sign_data_grad
    
    # Add clipping to maintain the original data range (e.g., [-1, 1])
    perturbed_images = torch.clamp(perturbed_images, -1, 1)
    
    # Return the perturbed images
    return perturbed_images


# --- 7. Evaluation Loop ---
def evaluate_model(model, data_loader, epsilon):
    """
    Evaluates the model on both clean and adversarial examples.
    """
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    loss_fn = nn.CrossEntropyLoss()

    # Store some examples for visualization
    clean_examples = []
    adv_examples = []
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # --- Clean Image Evaluation ---
        outputs_clean = model(images)
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        total += labels.size(0)
        correct_clean += (predicted_clean == labels).sum().item()

        # --- Adversarial Image Generation and Evaluation ---
        perturbed_images = fgsm_attack(model, loss_fn, images, labels, epsilon)
        outputs_adv = model(perturbed_images)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        correct_adv += (predicted_adv == labels).sum().item()

        # Save some examples for visualization if needed
        if len(clean_examples) < 5:
            # Store original image, its label, and the model's prediction
            clean_ex = (images[0], labels[0], predicted_clean[0])
            # Store perturbed image, its true label, and the model's new prediction
            adv_ex = (perturbed_images[0], labels[0], predicted_adv[0])
            clean_examples.append(clean_ex)
            adv_examples.append(adv_ex)
            
    clean_accuracy = 100 * correct_clean / total
    adv_accuracy = 100 * correct_adv / total

    print(f"\nEpsilon: {epsilon}")
    print(f"Accuracy on Clean Test Images: {clean_accuracy:.2f}% ({correct_clean}/{total})")
    print(f"Accuracy on Adversarial Test Images: {adv_accuracy:.2f}% ({correct_adv}/{total})")
    
    return clean_accuracy, adv_accuracy, clean_examples, adv_examples

# --- 8. Run Evaluation and Visualize Results ---
# Epsilon is the perturbation magnitude. Higher epsilon means more noticeable
# changes to the image but a stronger attack.
epsilon = 0.5 
clean_acc, adv_acc, clean_ex, adv_ex = evaluate_model(model, test_loader, epsilon)

# --- Visualization ---
print("\nVisualizing examples...")

# Function to de-normalize and show an image
def imshow(tensor, title=None):
    # The ToTensor transform moves image to [0,1] and Normalize moves it to [-1, 1]
    # We need to reverse this process:
    # 1. De-normalize: (tensor * std) + mean
    # 2. Convert to numpy and transpose for matplotlib
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the batch dimension
    image = image * 0.5 + 0.5     # de-normalize from [-1, 1] to [0, 1]
    npimg = image.detach().numpy()
    plt.imshow(npimg, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])

# Plot several examples
cnt = 0
plt.figure(figsize=(10, 8))
for i in range(len(clean_ex)):
    cnt += 1
    # Original image
    plt.subplot(len(clean_ex), 2, cnt)
    plt.ylabel(f"Example {i+1}", fontsize=14)
    orig_img, orig_label, orig_pred = clean_ex[i]
    imshow(orig_img, f"Original\nTrue: {class_names[orig_label]} | Pred: {class_names[orig_pred]}")

    cnt += 1
    # Adversarial image
    plt.subplot(len(clean_ex), 2, cnt)
    adv_img, adv_label, adv_pred = adv_ex[i]
    imshow(adv_img, f"Adversarial (eps={epsilon})\nTrue: {class_names[adv_label]} | Pred: {class_names[adv_pred]}")

plt.tight_layout()
plt.show()