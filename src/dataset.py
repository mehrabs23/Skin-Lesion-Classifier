# src/dataset.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(img_size=224, train=True):
    """
    Returns transform pipeline with/without augmentation.

    Args:
        img_size (int): Resize image to (img_size x img_size)
        train (bool): Apply augmentation if True

    Returns:
        transform (torchvision.transforms.Compose)
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(train_dir, val_dir, img_size, batch_size, num_workers):
    """
    Loads ImageFolder datasets and returns dataloaders.

    Args:
        train_dir (str): Path to training images
        val_dir (str): Path to validation images
        img_size (int): Resize target size
        batch_size (int): Mini-batch size
        num_workers (int): DataLoader parallel workers

    Returns:
        train_loader, val_loader, class_names
    """
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, train=True))
    val_ds   = datasets.ImageFolder(val_dir,   transform=build_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes