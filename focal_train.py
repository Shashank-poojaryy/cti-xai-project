"""
F1-improvement experiment: retrain the 3 models with FOCAL LOSS and a single
(milder) imbalance correction, instead of weighted-BCE(pos_weight=21) + sampler.

Rationale: the old setup applied imbalance correction TWICE (pos_weight AND an
oversampler), pushing the decision boundary so far toward 'positive' that
precision -> F1 collapsed. Focal loss focuses on hard examples and, with a
balanced sampler alone, yields a better-calibrated, higher-precision model.

Saves to models_v2/ (does NOT touch the current models/). Prints a head-to-head
comparison: AUC, Average Precision (the F1 ceiling), F1@0.5, and best-F1.
"""
import os, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models as tvm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from fast_dataloader import get_fast_dataloaders

OLD_DIR = r"C:\Users\NMAMIT\cti_project\models"
NEW_DIR = r"C:\Users\NMAMIT\cti_project\models_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(NEW_DIR, exist_ok=True)
MAX_EPOCHS, PHASE2, PATIENCE = 50, 6, 7
ALPHA, GAMMA = 0.6, 2.0          # focal: mild positive weight + hard-example focus


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, y):
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        at = self.a * y + (1 - self.a) * (1 - y)
        return (at * (1 - pt) ** self.g * ce).mean()


def build(name):
    if name == "densenet121":
        m = tvm.densenet121(weights="IMAGENET1K_V1"); m.classifier = nn.Linear(m.classifier.in_features, 1)
        head = m.classifier; blocks = [m.features.denseblock3, m.features.transition3, m.features.denseblock4, m.features.norm5]
    elif name == "resnet50":
        m = tvm.resnet50(weights="IMAGENET1K_V1"); m.fc = nn.Linear(m.fc.in_features, 1)
        head = m.fc; blocks = [m.layer3, m.layer4]
    else:
        m = tvm.efficientnet_b4(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        head = m.classifier; blocks = [m.features[-2], m.features[-1]]
    return m.to(DEVICE), head, blocks


def set_phase(m, head, blocks, phase):
    for p in m.parameters(): p.requires_grad = False
    for p in head.parameters(): p.requires_grad = True
    if phase == 2:
        for b in blocks:
            for p in b.parameters(): p.requires_grad = True


@torch.no_grad()
def infer(model, loader):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with torch.amp.autocast("cuda"): out = model(x)
        ps.extend(torch.sigmoid(out).float().cpu().numpy().ravel()); ys.extend(y.numpy().ravel())
    return np.array(ys), np.array(ps)


def best_f1(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s)); return f1s[i], grid[i]


def train_one(name, tr, va, te):
    print(f"\n=== {name} (focal alpha={ALPHA} gamma={GAMMA}) ===")
    m, head, blocks = build(name)
    crit = FocalLoss(ALPHA, GAMMA)
    scaler = torch.amp.GradScaler("cuda")
    set_phase(m, head, blocks, 1)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    sch = CosineAnnealingWarmRestarts(opt, T_0=10)
    best_auc, ctr, path = 0.0, 0, os.path.join(NEW_DIR, f"{name}_best.pth"); hist = []
    for ep in range(1, MAX_EPOCHS + 1):
        if ep == PHASE2:
            set_phase(m, head, blocks, 2)
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-5)
            sch = CosineAnnealingWarmRestarts(opt, T_0=10)
        m.train(); t0 = time.time()
        for x, y in tr:
            x = x.to(DEVICE); y = y.to(DEVICE).unsqueeze(1); opt.zero_grad()
            with torch.amp.autocast("cuda"): loss = crit(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        m.eval(); yv, pv = infer(m, va); vauc = roc_auc_score(yv, pv)
        hist.append({"epoch": ep, "val_auc": round(vauc, 4)})
        flag = ""
        if vauc > best_auc: best_auc, ctr = vauc, 0; torch.save(m.state_dict(), path); flag = " *best"
        else: ctr += 1
        print(f"  ep{ep:02d} [{time.time()-t0:.0f}s] val_auc {vauc:.4f}{flag}")
        if ctr >= PATIENCE: print(f"  early stop @ {ep}"); break
    pd.DataFrame(hist).to_csv(os.path.join(NEW_DIR, f"{name}_history.csv"), index=False)
    m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
    yt, pt = infer(m, te)
    auc, ap = roc_auc_score(yt, pt), average_precision_score(yt, pt)
    f1_05 = f1_score(yt, (pt >= 0.5).astype(int), zero_division=0)
    bf1, bthr = best_f1(yt, pt)
    res = {"model": name, "AUC": round(auc, 4), "AP": round(ap, 4),
           "F1@0.5": round(f1_05, 4), "bestF1": round(bf1, 4), "bestF1_thr": round(bthr, 4)}
    json.dump(res, open(os.path.join(NEW_DIR, f"{name}_v2_metrics.json"), "w"), indent=2)
    print(f"  -> AUC {auc:.4f} AP {ap:.4f} F1@0.5 {f1_05:.3f} bestF1 {bf1:.3f} @thr {bthr:.2f}")
    return res


@torch.no_grad()
def eval_old(name, te):
    m, _, _ = build(name)
    m.load_state_dict(torch.load(os.path.join(OLD_DIR, f"{name}_best.pth"), map_location=DEVICE)); m.eval()
    yt, pt = infer(m, te)
    return {"model": name, "AUC": round(roc_auc_score(yt, pt), 4), "AP": round(average_precision_score(yt, pt), 4),
            "F1@0.5": round(f1_score(yt, (pt >= 0.5).astype(int), zero_division=0), 4),
            "bestF1": round(best_f1(yt, pt)[0], 4)}


if __name__ == "__main__":
    tr, va, te = get_fast_dataloaders(batch_size=32)
    old, new = [], []
    for n in ["densenet121", "resnet50", "efficientnet_b4"]:
        old.append(eval_old(n, te))
        new.append(train_one(n, tr, va, te))
    print("\n" + "=" * 70 + "\nOLD (weighted-BCE + sampler)  vs  NEW (focal loss)\n" + "=" * 70)
    o = pd.DataFrame(old).add_suffix("_old").rename(columns={"model_old": "model"})
    nw = pd.DataFrame(new).add_suffix("_new").rename(columns={"model_new": "model"})
    comp = o.merge(nw, on="model")
    cols = ["model", "AUC_old", "AUC_new", "AP_old", "AP_new", "F1@0.5_old", "F1@0.5_new", "bestF1_old", "bestF1_new"]
    print(comp[cols].to_string(index=False))
    comp.to_csv(os.path.join(NEW_DIR, "comparison_old_vs_new.csv"), index=False)
    print("\nSaved models_v2/ + comparison_old_vs_new.csv (current models/ untouched)")
