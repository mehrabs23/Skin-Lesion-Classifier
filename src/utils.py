# src/utils.py

import os
import random
import numpy as np
import torch
import yaml
from collections import Counter


def load_config(path="config.yaml"):
    """
    Loads a YAML config file.

    Args:
        path (str): Path to config file

    Returns:
        dict: Parsed configuration
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(imagefolder_dataset, class_names):
    """
    Compute inverse-frequency class weights.

    Args:
        imagefolder_dataset: PyTorch ImageFolder dataset
        class_names (List[str]): Names of classes in order

    Returns:
        torch.Tensor: Normalized class weights
        List[int]: Sample counts per class
    """
    counts = Counter(imagefolder_dataset.targets)
    totals = [counts[i] for i in range(len(class_names))]
    inv = [0 if c == 0 else 1.0 / c for c in totals]
    weights = torch.tensor(inv, dtype=torch.float)
    weights = weights * (len(inv) / (weights.sum() + 1e-8))  # normalize
    return weights, totals
