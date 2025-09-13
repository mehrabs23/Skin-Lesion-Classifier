# src/split_dataset.py
import os
import random
import shutil
from pathlib import Path

random.seed(42)

source_dir = Path("data/raw")
dest_dir = Path("data")
splits = {"train": 0.7, "val": 0.2, "test": 0.1}

# Ensure target dirs exist
for split in splits:
    for class_dir in source_dir.iterdir():
        target_dir = dest_dir / split / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

# Split and copy files
for class_dir in source_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * splits["train"])
        n_val = int(n_total * splits["val"])
        split_data = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, image_list in split_data.items():
            for img_path in image_list:
                dest_path = dest_dir / split / class_dir.name / img_path.name
                shutil.copy(img_path, dest_path)

print("✅ Done splitting dataset into train, val, test.")
