"""Evaluate accuracy / AUC / sensitivity / specificity for each trained model.

Uses a balanced test subset (all test-set Cardiomegaly + equal Normal) so
accuracy is meaningful (the full test set is ~96% Normal, which would inflate
accuracy regardless of skill). Inference is batched on the GPU.
"""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from generate_xai import load_model, transform, IMAGE_DIR, DATA_DIR, DEVICE
import warnings
warnings.filterwarnings("ignore")

MODELS = ["densenet121", "resnet50", "efficientnet_b4"]
BATCH = 64


def get_eval_set():
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sample(n=len(pos), random_state=42)
    sub = pd.concat([pos, neg]).reset_index(drop=True)
    return sub["Image Index"].values, sub["label"].values.astype(int)


@torch.no_grad()
def predict(model, names):
    probs = []
    for i in range(0, len(names), BATCH):
        tensors = [transform(Image.open(os.path.join(IMAGE_DIR, n)).convert("RGB"))
                   for n in names[i:i + BATCH]]
        x = torch.stack(tensors).to(DEVICE)
        logits = model(x).squeeze(1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs)


def main():
    names, labels = get_eval_set()
    print(f"\nBalanced test set: {int(labels.sum())} Cardiomegaly + "
          f"{int((labels == 0).sum())} Normal = {len(labels)} images\n")
    print(f"{'Model':16s} {'Accuracy':>9s} {'AUC':>7s} {'Sensitivity':>12s} {'Specificity':>12s}")
    print("-" * 60)
    for m in MODELS:
        model, _ = load_model(m)
        probs = predict(model, names)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        print(f"{m:16s} {acc*100:7.2f}%  {auc:6.3f}  {sens*100:10.2f}%  {spec*100:10.2f}%")


if __name__ == "__main__":
    main()
