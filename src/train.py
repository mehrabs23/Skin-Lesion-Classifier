# src/train.py
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd  # Added for CSV output
import matplotlib.pyplot as plt  # Added for F1 bar chart

from src.dataset import get_dataloaders  # type: ignore
from src.utils import load_config, set_seed, compute_class_weights, calculate_metrics, plot_confusion_matrix  # type: ignore
from model.model import get_model, save_checkpoint  # type: ignore
from torchvision import datasets, transforms

def validate(model, loader, device, class_names, epoch=None, output_dir=None):
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
    # Compute detailed metrics
    precision, recall, f1, support = calculate_metrics(y_true, y_pred, class_names)
    print("\nValidation Metrics:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}, Support={support[i]}")
    print(f"  Overall Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    # Save metrics & confusion matrix if enabled
    if output_dir and epoch:
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f"confmat_epoch{epoch}.png")
        plot_confusion_matrix(y_true, y_pred, class_names, output_path=cm_path)

        # Save metrics CSV
        metrics_df = pd.DataFrame({
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support
        })
        metrics_path = os.path.join(output_dir, f"metrics_epoch{epoch}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"📄 Metrics saved to {metrics_path}")


        # Save bar chart of F1 scores
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(class_names, f1, color='skyblue')
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 Scores by Class (Epoch {epoch})")
        ax.set_ylim(0, 1.0)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f1_bar_epoch{epoch}.png"))
        plt.close()
        print(f"📊 F1 bar chart saved to f1_bar_epoch{epoch}.png")

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

    print("Run ID: exp-002")
    print("Model: ResNet50")
    print("Layers unfrozen: layer3, layer4, fc")
    print("LR: 1e-4 | Epochs: 10")

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
        val_loss, val_acc = validate(model, val_loader, device, classes, epoch, output_dir=cfg["eval"]["outputs_dir"])

        if sched:
            sched.step()

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