"""
Extract the Cardiomegaly subset of BBox_List_2017.csv into a dedicated file.

BBox_List_2017.csv has 984 annotations across 8 pathologies; only the
Cardiomegaly rows are the XAI/CTI target. This writes that subset explicitly
(does NOT modify the original) and reports the full breakdown + split coverage.

Output: data/BBox_Cardiomegaly.csv
"""
import os
import pandas as pd

DATA_DIR  = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR = r"C:\Users\NMAMIT\cti_project\images"

bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
print(f"BBox_List_2017.csv total annotations: {len(bb)}")
print("\nBreakdown by Finding Label:")
print(bb["Finding Label"].value_counts().to_string())

cardio = bb[bb["Finding Label"] == "Cardiomegaly"].copy()
cardio = cardio.drop_duplicates("Image Index")

# coverage checks
avail = set(os.listdir(IMAGE_DIR))
cardio["on_disk"] = cardio["Image Index"].isin(avail)
for split in ["train", "val", "test"]:
    ids = set(pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))["Image Index"])
    cardio[f"in_{split}"] = cardio["Image Index"].isin(ids)

out_cols = ["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"]
cardio[out_cols].to_csv(os.path.join(DATA_DIR, "BBox_Cardiomegaly.csv"), index=False)

print(f"\nCardiomegaly bbox annotations: {len(cardio)}")
print(f"  on disk:        {int(cardio['on_disk'].sum())}/{len(cardio)}")
print(f"  in train split: {int(cardio['in_train'].sum())}")
print(f"  in val split:   {int(cardio['in_val'].sum())}")
print(f"  in test split:  {int(cardio['in_test'].sum())}")
print(f"\nSaved -> data/BBox_Cardiomegaly.csv ({len(cardio)} rows, columns: {out_cols})")
print("Original BBox_List_2017.csv left untouched.")
