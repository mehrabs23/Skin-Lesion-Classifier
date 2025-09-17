# src/predict.py


import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse


from src.utils import load_config
from src.model import get_model  # ✅ IMPORTANT: use same model builder used in training

def load_image(img_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: [1, C, H, W]




def predict(img_path, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load model with same architecture
    model = get_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)


    # Load weights
    ckpt_path = "models/resnet18_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()


    # Load and preprocess image
    img = load_image(img_path, cfg["data"]["img_size"]).to(device)


    # Predict
    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class_idx].item()


    class_name = cfg["data"]["class_names"][pred_class_idx]
    print(f"✅ Prediction: {class_name} ({pred_prob * 100:.2f}%)")


    # Optional: Show all class probabilities
    print("\n📊 Class Probabilities:")
    for i, cls in enumerate(cfg["data"]["class_names"]):
        print(f"  {cls}: {probs[0, i].item():.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    args = parser.parse_args()


    cfg = load_config("config.yaml")
    predict(args.img, cfg)
