"""Quick sanity check: load each trained .pth and verify AUC on 50 test images.

Confirms the weights load and behave correctly on this PC (RTX 4070, cu128).
Uses a balanced 25 Cardiomegaly + 25 Normal subset so AUC is well-defined.
"""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from generate_xai import load_model, transform, IMAGE_DIR, DATA_DIR, DEVICE
import warnings
warnings.filterwarnings("ignore")

N_PER_CLASS = 25
MODELS   = ["densenet121", "resnet50", "efficientnet_b4"]
EXPECTED = {"densenet121": 0.88, "resnet50": 0.89, "efficientnet_b4": 0.86}


def sample_images():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    pos = test_df[test_df["label"] == 1].sample(n=N_PER_CLASS, random_state=42)
    neg = test_df[test_df["label"] == 0].sample(n=N_PER_CLASS, random_state=42)
    sub = pd.concat([pos, neg]).reset_index(drop=True)
    return sub["Image Index"].values, sub["label"].values


@torch.no_grad()
def predict(model, img_names):
    probs = []
    for name in img_names:
        img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        logit = model(x).squeeze()
        probs.append(torch.sigmoid(logit).item())
    return np.array(probs)


def main():
    names, labels = sample_images()
    print(f"\nSanity AUC on {len(names)} test images "
          f"({N_PER_CLASS} Cardiomegaly + {N_PER_CLASS} Normal)\n")
    for model_name in MODELS:
        model, _ = load_model(model_name)
        probs = predict(model, names)
        auc = roc_auc_score(labels, probs)
        exp = EXPECTED[model_name]
        flag = "OK" if abs(auc - exp) <= 0.10 else "CHECK"
        print(f"{model_name:16s} AUC={auc:.4f}  (expected ~{exp:.2f})  [{flag}]")


if __name__ == "__main__":
    main()
