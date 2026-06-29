"""
Train DenseNet121 / ResNet50 / EfficientNet-B4 for Cardiomegaly vs Normal on the
patient-wise split (see patient_split.py). Q1/Q2 training protocol:

  - Two-phase fine-tuning:
      Phase 1 (epochs 1-5): freeze backbone, train classifier head only (lr=1e-4)
      Phase 2 (epoch 6+)  : unfreeze last 2 blocks, fine-tune end-to-end (lr=1e-5)
  - BCEWithLogitsLoss(pos_weight) + WeightedRandomSampler (both, per spec)
  - Adam(weight_decay=1e-5), CosineAnnealingWarmRestarts(T_0=10)
  - Early stopping patience=7 on val AUC-ROC, max 50 epochs
  - AMP (mixed precision) on GPU for speed
  - Per-epoch checkpoint (resume-safe) + full history CSV + test-metrics JSON

Reads the pre-resized memmap cache via fast_dataloader (GPU-bound throughput).
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from fast_dataloader import get_fast_dataloaders

DATA_DIR   = r"C:\Users\NMAMIT\cti_project\data"
MODELS_DIR = r"C:\Users\NMAMIT\cti_project\models"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 32
MAX_EPOCHS   = 50
PHASE2_EPOCH = 6           # unfreeze last 2 blocks from this epoch
LR_HEAD      = 1e-4
LR_FINETUNE  = 1e-5
PATIENCE     = 7
WEIGHT_DECAY = 1e-5
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── MODEL ────────────────────────────────────────────────────────────────
def build_model(name):
    if name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "efficientnet_b4":
        m = models.efficientnet_b4(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def head_module(model, name):
    if name == "densenet121":   return model.classifier
    if name == "resnet50":      return model.fc
    return model.classifier  # efficientnet


def last_blocks(model, name):
    """The 'last 2 blocks' unfrozen in phase 2 (plus following norm)."""
    if name == "densenet121":
        return [model.features.denseblock3, model.features.transition3,
                model.features.denseblock4, model.features.norm5]
    if name == "resnet50":
        return [model.layer3, model.layer4]
    return [model.features[-2], model.features[-1]]  # efficientnet


def set_phase(model, name, phase):
    for p in model.parameters():
        p.requires_grad = False
    for p in head_module(model, name).parameters():
        p.requires_grad = True
    if phase == 2:
        for blk in last_blocks(model, name):
            for p in blk.parameters():
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  phase {phase}: trainable {trainable:,}/{total:,} params")


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)


# ─── EPOCH LOOPS ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, train):
    model.train() if train else model.eval()
    total_loss, labels_all, probs_all = 0.0, [], []
    torch.set_grad_enabled(train)
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE).unsqueeze(1)
        if train:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            with torch.amp.autocast("cuda"):
                out = model(x); loss = criterion(out, y)
        total_loss += loss.item()
        labels_all.extend(y.detach().cpu().numpy().ravel())
        probs_all.extend(torch.sigmoid(out).detach().float().cpu().numpy().ravel())
    torch.set_grad_enabled(True)
    auc = roc_auc_score(labels_all, probs_all)
    return total_loss / len(loader), auc, np.array(labels_all), np.array(probs_all)


def test_metrics(model, loader):
    _, auc, y, p = run_epoch(model, loader, nn.BCEWithLogitsLoss(),
                             None, None, train=False)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auc_roc": float(auc),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ─── TRAIN ONE MODEL ──────────────────────────────────────────────────────
def train_model(name, train_loader, val_loader, test_loader, pos_weight):
    print(f"\n{'='*60}\nTraining {name.upper()}\n{'='*60}")
    model = build_model(name)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    scaler = torch.amp.GradScaler("cuda")

    set_phase(model, name, 1)
    optimizer = make_optimizer(model, LR_HEAD)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_auc, patience_ctr, best_path = 0.0, 0, os.path.join(MODELS_DIR, f"{name}_best.pth")
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == PHASE2_EPOCH:           # switch to fine-tuning phase
            print("  --> Phase 2: unfreezing last 2 blocks, lr=1e-5")
            set_phase(model, name, 2)
            optimizer = make_optimizer(model, LR_FINETUNE)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

        t0 = time.time()
        tr_loss, tr_auc, _, _ = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, train=True)
        vl_loss, vl_auc, _, _ = run_epoch(model, val_loader, criterion,
                                          None, None, train=False)
        scheduler.step()
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_auc": round(tr_auc, 4), "val_loss": round(vl_loss, 4),
                        "val_auc": round(vl_auc, 4)})
        pd.DataFrame(history).to_csv(
            os.path.join(MODELS_DIR, f"{name}_history.csv"), index=False)

        flag = ""
        if vl_auc > best_auc:
            best_auc, patience_ctr = vl_auc, 0
            torch.save(model.state_dict(), best_path)
            flag = "  <-- best saved"
        else:
            patience_ctr += 1
        print(f"  Epoch {epoch:02d} [{dt:.0f}s] train_loss {tr_loss:.4f} "
              f"train_auc {tr_auc:.4f} | val_loss {vl_loss:.4f} val_auc {vl_auc:.4f}{flag}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best val AUC {best_auc:.4f})")
            break

    # ── Test-set metrics on best checkpoint ──
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    metrics = test_metrics(model, test_loader)
    metrics["best_val_auc"] = float(best_auc)
    with open(os.path.join(MODELS_DIR, f"{name}_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    gate = "PASS" if metrics["auc_roc"] >= 0.80 else "FAIL (<0.80)"
    print(f"  TEST AUC {metrics['auc_roc']:.4f} | Sens {metrics['recall_sensitivity']:.3f} "
          f"| Spec {metrics['specificity']:.3f} | BalAcc {metrics['balanced_accuracy']:.3f}  [{gate}]")
    return metrics


# ─── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["densenet121", "resnet50", "efficientnet_b4"])
    args = ap.parse_args()

    print("Device:", DEVICE)
    train_loader, val_loader, test_loader = get_fast_dataloaders(BATCH_SIZE)
    labels = train_loader.dataset.labels.astype(int)
    pos_weight = (labels == 0).sum() / (labels == 1).sum()
    print(f"pos_weight = {pos_weight:.2f}")

    summary = {}
    for name in args.models:
        summary[name] = train_model(name, train_loader, val_loader, test_loader, pos_weight)

    print("\n" + "=" * 60 + "\nFINAL TEST METRICS\n" + "=" * 60)
    for name, m in summary.items():
        print(f"{name:16s} AUC {m['auc_roc']:.4f} | F1 {m['f1']:.3f} | "
              f"Sens {m['recall_sensitivity']:.3f} | Spec {m['specificity']:.3f} | "
              f"BalAcc {m['balanced_accuracy']:.3f}")
