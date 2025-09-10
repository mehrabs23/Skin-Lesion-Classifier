# src/evaluate.py

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils import load_config 
from model.model import load_checkpoint


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text labels inside matrix
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(cfg["data"]["test_dir"], transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = load_checkpoint(
        os.path.join(cfg["train"]["checkpoint_dir"], cfg["train"]["checkpoint_name"]),
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        device=device
    )

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # Save classification report
    os.makedirs(cfg["eval"]["outputs_dir"], exist_ok=True)
    report_path = os.path.join(cfg["eval"]["outputs_dir"], cfg["eval"]["report_txt_name"])
    report = classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4)
    print(report)
    with open(report_path, "w") as f:
        f.write(report)

    # Save confusion matrix
    if cfg["eval"]["save_cm_png"]:
        cm = confusion_matrix(y_true, y_pred)
        cm_path = os.path.join(cfg["eval"]["outputs_dir"], cfg["eval"]["cm_png_name"])
        plot_confusion_matrix(cm, test_ds.classes, cm_path)
        print(f"✅ Confusion matrix saved to: {cm_path}")
