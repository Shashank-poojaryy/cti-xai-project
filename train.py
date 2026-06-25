import os
import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from dataloader import get_dataloaders

# ─── CONFIG ───────────────────────────────────────────────
DATA_DIR   = r'C:\Users\NMAMIT\cti_project\data'
IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'
MODELS_DIR = r'C:\Users\NMAMIT\cti_project\models'
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 1e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)


# ─── MODEL BUILDER ────────────────────────────────────────
def build_model(model_name):
    if model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    return model.to(DEVICE)


# ─── TRAIN ONE EPOCH ──────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc


# ─── EVALUATE ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc


# ─── MAIN TRAINING LOOP ───────────────────────────────────
def train_model(model_name):
    print(f"\n{'='*50}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*50}")

    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, IMAGE_DIR, BATCH_SIZE)

    model = build_model(model_name)

    # Class weight for imbalance
    pos_weight = torch.tensor([21.7]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_auc = 0
    patience = 5
    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc     = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_auc': round(train_auc, 4),
            'val_loss': round(val_loss, 4),
            'val_auc': round(val_auc, 4)
        })

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
            print(f"  --> Best model saved (Val AUC: {best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(MODELS_DIR, f'{model_name}_history.csv'), index=False)
    print(f"\nBest Val AUC for {model_name}: {best_auc:.4f}")
    return best_auc


# ─── RUN ALL 3 MODELS ─────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for model_name in ['densenet121', 'resnet50', 'efficientnet_b4']:
        results[model_name] = train_model(model_name)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for name, auc in results.items():
        print(f"{name}: Best Val AUC = {auc:.4f}")