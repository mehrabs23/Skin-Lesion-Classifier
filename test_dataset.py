from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Path to your cleaned test directory
test_path = '/home/mehrab/skin-lesion-classifier/data/test'

# Define the transform (you can adjust image size if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the dataset
test_dataset = ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Print some info
print("Number of test samples:", len(test_dataset))
print("Number of classes:", len(test_dataset.classes))
print("Class names:", test_dataset.classes)

# Inspect a single batch
for images, labels in test_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break
