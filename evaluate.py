"""
Step 3 (Table 1) - test-set classification performance for the 3 trained models.

Metrics on the held-out patient-wise test set:
  AUC-ROC with 95% bootstrap CI (n=1000), F1, Precision, Recall/Sensitivity,
  Specificity, Balanced Accuracy, confusion matrix.
Outputs:
  results/classification_performance.csv   (Table 1)
  results/roc_curves.png/.pdf              (Figure 7: ROC + PR, 300 DPI)
"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
from fast_dataloader import get_fast_dataloaders
from xai_core import load_trained  # not used for arch; build below instead
from torchvision import models as tvm

MODELS_DIR  = r"C:\Users\NMAMIT\cti_project\models"
RESULTS_DIR = r"C:\Users\NMAMIT\cti_project\results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["densenet121", "resnet50", "efficientnet_b4"]
LABELNICE = {"densenet121": "DenseNet121", "resnet50": "ResNet50",
             "efficientnet_b4": "EfficientNet-B4"}
os.makedirs(RESULTS_DIR, exist_ok=True)


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
        ps.extend(torch.sigmoid(out).float().cpu().numpy().ravel())
        ys.extend(y.numpy().ravel())
    return np.array(ys), np.array(ps)


def bootstrap_auc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def main():
    _, _, test_loader = get_fast_dataloaders(batch_size=64)
    rows, curves = [], {}
    for name in MODELS:
        model = build(name)
        y, p = predict(model, test_loader)
        pred = (p >= 0.5).astype(int)
        auc = roc_auc_score(y, p)
        lo, hi = bootstrap_auc(y, p)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "Model": LABELNICE[name],
            "AUC-ROC": round(auc, 4),
            "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(sens, 4),
            "Specificity": round(spec, 4),
            "Balanced Acc": round((sens + spec) / 2, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
        curves[name] = (y, p, auc)
        print(f"{LABELNICE[name]:16s} AUC {auc:.4f} CI[{lo:.3f},{hi:.3f}] "
              f"Sens {sens:.3f} Spec {spec:.3f}")
        del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "classification_performance.csv"), index=False)
    print("\nSaved classification_performance.csv")

    # ── Figure 7: ROC + PR ──
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name in MODELS:
        y, p, auc = curves[name]
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, lw=2, label=f"{LABELNICE[name]} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax2.plot(rec, prec, lw=2, label=f"{LABELNICE[name]} (AP={ap:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate", fontsize=12); ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curves (Test Set)", fontsize=14); ax1.legend(fontsize=11)
    ax2.set_xlabel("Recall", fontsize=12); ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curves (Test Set)", fontsize=14); ax2.legend(fontsize=11)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(RESULTS_DIR, f"roc_curves.{ext}"), dpi=300, bbox_inches="tight")
    print("Saved roc_curves.png/.pdf")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
