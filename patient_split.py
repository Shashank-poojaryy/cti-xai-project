"""
Patient-wise 70/10/20 train/val/test split for NIH ChestX-ray14
(Cardiomegaly vs No-Finding).

Q1/Q2 requirement: NIH has multiple X-rays per patient. A random image-level
split leaks patients across train/test (the previous split had 70.9% test-image
patient overlap). This script splits at the PATIENT level so that every image of
a given patient lands in exactly one split -> ZERO patient leakage.

Output: data/train.csv, data/val.csv, data/test.csv
Columns: full Data_Entry columns + 'label' (1=Cardiomegaly, 0=Normal).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR = r"C:\Users\NMAMIT\cti_project\images"
SEED = 42

# ── Load and filter ────────────────────────────────────────────────────────
data = pd.read_csv(os.path.join(DATA_DIR, "Data_Entry_2017.csv"))

cardio = data[data["Finding Labels"].str.contains("Cardiomegaly")].copy()
normal = data[data["Finding Labels"] == "No Finding"].copy()
cardio["label"] = 1
normal["label"] = 0

df = pd.concat([cardio, normal], ignore_index=True)
print(f"Total candidate images: {len(df)}  |  Cardiomegaly: {len(cardio)}  Normal: {len(normal)}")

# Keep only images that actually exist on disk (avoids training crashes on
# missing files). Done BEFORE the split so the split stays patient-grouped.
available = set(os.listdir(IMAGE_DIR))
before = len(df)
df = df[df["Image Index"].isin(available)].reset_index(drop=True)
print(f"On disk: {len(df)} images (dropped {before - len(df)} missing)  "
      f"| Cardiomegaly: {int((df['label']==1).sum())}  Normal: {int((df['label']==0).sum())}")

# ── Patient-level table (a patient is 'positive' if ANY of their images is
#    Cardiomegaly) -> used to stratify the patient split ─────────────────────
patient_label = df.groupby("Patient ID")["label"].max().reset_index()
patient_label.columns = ["Patient ID", "patient_pos"]
n_patients = len(patient_label)
print(f"Unique patients: {n_patients}  |  positive patients: {int(patient_label['patient_pos'].sum())}")

# ── Split PATIENT IDs 70 / 10 / 20, stratified on patient positivity ────────
train_p, temp_p = train_test_split(
    patient_label, test_size=0.30,
    stratify=patient_label["patient_pos"], random_state=SEED)
val_p, test_p = train_test_split(
    temp_p, test_size=0.667,
    stratify=temp_p["patient_pos"], random_state=SEED)

train_ids = set(train_p["Patient ID"])
val_ids   = set(val_p["Patient ID"])
test_ids  = set(test_p["Patient ID"])

# ── Assign every image to its patient's split ──────────────────────────────
train = df[df["Patient ID"].isin(train_ids)].reset_index(drop=True)
val   = df[df["Patient ID"].isin(val_ids)].reset_index(drop=True)
test  = df[df["Patient ID"].isin(test_ids)].reset_index(drop=True)

# ── Verify ZERO patient overlap (must pass) ────────────────────────────────
assert len(train_ids & test_ids) == 0, "DATA LEAKAGE: train/test patient overlap"
assert len(train_ids & val_ids)  == 0, "DATA LEAKAGE: train/val patient overlap"
assert len(val_ids & test_ids)   == 0, "DATA LEAKAGE: val/test patient overlap"
# Sanity: every image accounted for, no duplicate images across splits
assert len(train) + len(val) + len(test) == len(df), "image count mismatch"
print("\nPatient-wise split verified - zero overlap confirmed")

# ── Save ───────────────────────────────────────────────────────────────────
train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

# ── Report ─────────────────────────────────────────────────────────────────
def summarize(name, d, ids):
    pos = int((d["label"] == 1).sum())
    neg = int((d["label"] == 0).sum())
    print(f"  {name:5s} | images={len(d):6d} | patients={len(ids):6d} "
          f"| Cardiomegaly={pos:5d} | Normal={neg:6d} | prevalence={pos/len(d)*100:.2f}%")

print("\nSplit summary:")
summarize("train", train, train_ids)
summarize("val",   val,   val_ids)
summarize("test",  test,  test_ids)

pos_weight = (train["label"] == 0).sum() / (train["label"] == 1).sum()
print(f"\nTrain pos_weight = neg/pos = {pos_weight:.2f}")
print("Saved train.csv / val.csv / test.csv to data/")
