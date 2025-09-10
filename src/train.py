# src/train.py

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.dataset import get_dataloaders
from src.utils import load_config, set_seed, compute_class_weights
from model.model import get_model, save_checkpoint
from torchvision import datasets, transforms


def validate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            val_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
    val_loss /= len(loader.dataset)
    val_acc = accuracy_score(y_true, y_pred)
    return val_loss, val_acc


def main():
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["eval"]["outputs_dir"], exist_ok=True)

    # Data
    train_loader, val_loader, classes = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        cfg["data"]["img_size"], cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    )

    # Class weights (optional)
    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=transforms.ToTensor())
    class_weights, counts = compute_class_weights(train_ds, classes)

    # Model
    model = get_model(cfg["model"]["name"], num_classes=cfg["model"]["num_classes"],
                      pretrained=cfg["model"]["pretrained"]).to(device)

    # Loss
    if cfg["train"]["use_class_weights"]:
        ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        ce = nn.CrossEntropyLoss()

    # Optimizer + LR scheduler
    opt = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = StepLR(opt, step_size=cfg["train"]["step_size"], gamma=cfg["train"]["gamma"]) if cfg["train"]["scheduler"] == "step" else None

    # Training loop
    best_val = float("inf")
    patience = cfg["train"]["early_stopping_patience"]
    no_improve = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = validate(model, val_loader, device)

        if sched: sched.step()

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping + checkpointing
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], cfg["train"]["checkpoint_name"])
            save_checkpoint(model, ckpt_path)
            print(f"✅ Saved best model to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️ Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
