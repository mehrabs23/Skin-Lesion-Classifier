# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="resnet50", num_classes=7, pretrained=True):
    """
    Load a pretrained CNN (resnet18 or resnet50) and replace the final layer.
    Matches saved checkpoint architecture (single linear fc layer).
    """
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Freeze all layers
    for param in m.parameters():
        param.requires_grad = False

    # Unfreeze last two blocks (layer3 and layer4)
    for param in m.layer3.parameters():
        param.requires_grad = True
    for param in m.layer4.parameters():
        param.requires_grad = True

    # Replace final classification layer with a single Linear layer
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)  # ✅ Match checkpoint architecture

    # Unfreeze the fc layer
    for param in m.fc.parameters():
        param.requires_grad = True

    return m


def save_checkpoint(model, path):
    """
    Save model weights to a file.
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(path, model_name="resnet50", num_classes=7, device="cpu"):
    """
    Load model from a saved checkpoint.
    """
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
