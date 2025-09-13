from dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

# Sanity check
for images, labels in train_loader:
    print(f"Batch size: {images.shape[0]}, Image shape: {images.shape}, Labels: {labels}")
    break
