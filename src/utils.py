# src/utils.py

import os
import random
import numpy as np
import torch
import yaml
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


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


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate precision, recall, f1, and support for each class.
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
    return precision, recall, f1, support


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Plot and save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
