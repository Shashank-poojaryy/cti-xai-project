"""
Operating-point analysis. The default threshold 0.5 gives poor specificity
because training used aggressive imbalance correction (pos_weight + sampler).
Here we pick the decision threshold on the VALIDATION set (no test leakage) by
two standard criteria and report the resulting TEST operating point:

  - Youden's J  (max sensitivity+specificity-1)  -> balanced screening point
  - F1-optimal  (max F1 on the minority class)

Saves results/classification_performance_tuned.csv. AUC is threshold-independent
and unchanged; only the operating point moves.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models as tvm
from sklearn.metrics import (roc_curve, f1_score, precision_score,
                             confusion_matrix, roc_auc_score)
from fast_dataloader import get_fast_dataloaders

MODELS_DIR  = r"C:\Users\NMAMIT\cti_project\models"
RESULTS_DIR = r"C:\Users\NMAMIT\cti_project\results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["densenet121", "resnet50", "efficientnet_b4"]
NICE = {"densenet121": "DenseNet121", "resnet50": "ResNet50", "efficientnet_b4": "EfficientNet-B4"}


def build(name):
    if name == "densenet121":
        m = tvm.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif name == "resnet50":
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        m = tvm.efficientnet_b4(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_best.pth"), map_location=DEVICE))
    return m.eval().to(DEVICE)


@torch.no_grad()
def predict(model, loader):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        ps.extend(torch.sigmoid(out).float().cpu().numpy().ravel()); ys.extend(y.numpy().ravel())
    return np.array(ys), np.array(ps)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"threshold": round(float(thr), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(sens, 4), "Specificity": round(spec, 4),
            "Balanced Acc": round((sens + spec) / 2, 4)}


def main():
    _, val_loader, test_loader = get_fast_dataloaders(batch_size=64)
    rows = []
    for name in MODELS:
        model = build(name)
        yv, pv = predict(model, val_loader)
        yt, pt = predict(model, test_loader)
        # Youden's J on validation
        fpr, tpr, thr = roc_curve(yv, pv)
        j_thr = thr[np.argmax(tpr - fpr)]
        # F1-optimal on validation
        grid = np.linspace(0.01, 0.99, 99)
        f1_thr = grid[np.argmax([f1_score(yv, (pv >= t).astype(int), zero_division=0) for t in grid])]
        auc = roc_auc_score(yt, pt)
        for crit, t in [("default 0.5", 0.5), ("Youden J", j_thr), ("F1-optimal", f1_thr)]:
            m = metrics_at(yt, pt, t)
            rows.append({"Model": NICE[name], "Criterion": crit, "AUC": round(auc, 4), **m})
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "classification_performance_tuned.csv"), index=False)
    print(df.to_string(index=False))
    print("\nSaved classification_performance_tuned.csv")


if __name__ == "__main__":
    main()
