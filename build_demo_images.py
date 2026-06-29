"""
Build a curated demo_images/ folder for the Streamlit app.

Picks HELD-OUT TEST-SPLIT images only (never seen in training):
  demo_images/cardiomegaly/  - test-split Cardiomegaly images WITH ground-truth
                               bounding boxes, ordered best-CTI-first (heatmap
                               lands on the heart -> good demo)
  demo_images/normal/        - random test-split Normal images
  demo_images/manifest.csv   - true label, patient id, split, bbox flag, CTI
  demo_images/README.txt     - what to upload and what to expect
"""
import os, shutil
import numpy as np
import pandas as pd

ROOT      = r"C:\Users\NMAMIT\cti_project"
DATA_DIR  = os.path.join(ROOT, "data")
IMAGE_DIR = os.path.join(ROOT, "images")
DEMO_DIR  = os.path.join(ROOT, "demo_images")
RESULTS   = os.path.join(ROOT, "results")
N_NORMAL  = 15
SEED      = 42

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_ids = set(test["Image Index"])
pid = test.set_index("Image Index")["Patient ID"]

bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
bb = bb[bb["Finding Label"] == "Cardiomegaly"].drop_duplicates("Image Index")
bbox_ids = set(bb["Image Index"])

# Cardiomegaly demo = test-split images that have a bbox, ordered by mean CTI
per = pd.read_csv(os.path.join(RESULTS, "cti_per_image.csv"))
mean_cti = per.groupby("image_id")["cti"].mean()
cardio = [i for i in test_ids if i in bbox_ids]
cardio = sorted(cardio, key=lambda f: mean_cti.get(f.replace(".png", ""), 0), reverse=True)

# Normal demo = random test-split Normal images
normal_pool = test[test["label"] == 0]["Image Index"].values
rng = np.random.default_rng(SEED)
normal = rng.choice(normal_pool, size=min(N_NORMAL, len(normal_pool)), replace=False)

# (re)build folders
for sub in ("cardiomegaly", "normal"):
    d = os.path.join(DEMO_DIR, sub)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

rows = []
for f in cardio:
    shutil.copy(os.path.join(IMAGE_DIR, f), os.path.join(DEMO_DIR, "cardiomegaly", f))
    rows.append({"filename": f, "true_label": "Cardiomegaly", "patient_id": int(pid[f]),
                 "split": "test", "has_bbox": True,
                 "mean_cti": round(float(mean_cti.get(f.replace(".png", ""), np.nan)), 4)})
for f in normal:
    shutil.copy(os.path.join(IMAGE_DIR, f), os.path.join(DEMO_DIR, "normal", f))
    rows.append({"filename": f, "true_label": "Normal", "patient_id": int(pid[f]),
                 "split": "test", "has_bbox": False, "mean_cti": np.nan})

man = pd.DataFrame(rows)
man.to_csv(os.path.join(DEMO_DIR, "manifest.csv"), index=False)

readme = f"""DEMO IMAGES for the XAI Reliability Streamlit app
==================================================
All images are from the HELD-OUT TEST split (patient-independent; the models
never saw these patients during training).

cardiomegaly/  {len(cardio)} confirmed Cardiomegaly images, each with a ground-truth
               bounding box. Files are ordered by mean CTI (see manifest.csv);
               the top ones give the cleanest heatmaps over the heart.
normal/        {len(normal)} confirmed "No Finding" (Normal) images.

HOW TO USE
1. In the app, upload any file from cardiomegaly/ (start with the highest-CTI ones).
2. Select a model (DenseNet121 = most trustworthy) and an XAI method
   (Grad-CAM++ ranked #1). Click Analyze.
3. The heatmap should highlight the cardiac silhouette for Cardiomegaly cases.

NOTE ON PREDICTIONS
The models are tuned for high sensitivity (they rarely miss cardiomegaly), so at
the default 0.5 threshold they OVER-predict positive: some Normal images may be
labelled "Cardiomegaly" with low confidence. That is expected from the class-
imbalance handling -- judge the heatmap quality, not just the label.

manifest.csv lists every demo image with its true label, patient id, and CTI.
"""
with open(os.path.join(DEMO_DIR, "README.txt"), "w") as fh:
    fh.write(readme)

print(f"demo_images/ built: {len(cardio)} cardiomegaly + {len(normal)} normal")
print("Top 5 cardiomegaly demo images (best heatmaps):")
print(man[man.true_label == "Cardiomegaly"].head(5)[["filename", "patient_id", "mean_cti"]].to_string(index=False))
