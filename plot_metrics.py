# plot_metrics.py

import json
import matplotlib.pyplot as plt

# Load metrics
import os

metrics_path = os.path.join(cfg["eval"]["outputs_dir"], "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_log, f, indent=2)
print(f"📈 Saved training metrics to {metrics_path}")

epochs = list(range(1, len(metrics["train_loss"]) + 1))

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker="o")
plt.plot(epochs, metrics["val_loss"], label="Val Loss", marker="o")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, metrics["val_acc"], label="Val Accuracy", color="green", marker="o")
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()

plt.show()
