# src/utils.py

import os
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix


def load_config(path="config.yaml"):
    """
    Loads a YAML config file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(imagefolder_dataset, class_names):
    """
    Compute inverse-frequency class weights.
    """
    counts = Counter(imagefolder_dataset.targets)
    totals = [counts[i] for i in range(len(class_names))]
    inv = [0 if c == 0 else 1.0 / c for c in totals]
    weights = torch.tensor(inv, dtype=torch.float)
    weights = weights * (len(inv) / (weights.sum() + 1e-8))  # normalize
    return weights, totals


def calculate_metrics(y_true, y_pred, target_names):
    """
    Print classification report and return as dict.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    return report


def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
