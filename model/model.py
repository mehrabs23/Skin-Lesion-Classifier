import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="resnet18", num_classes=7, pretrained=True):
    """
    Load a pretrained CNN (resnet18 or resnet50) and replace the final layer.

    Args:
        model_name (str): 'resnet18' or 'resnet50'
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load ImageNet weights

    Returns:
        nn.Module: The model
    """
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def save_checkpoint(model, path):
    """
    Save model weights to a file.
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(path, model_name="resnet18", num_classes=7, device="cpu"):
    """
    Load model from a saved checkpoint.

    Args:
        path (str): File path to model weights
        model_name (str): 'resnet18' or 'resnet50'
        num_classes (int): Number of output classes
        device (str): 'cpu' or 'cuda'

    Returns:
        nn.Module: The model with weights loaded
    """
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
