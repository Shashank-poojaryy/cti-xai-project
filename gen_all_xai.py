"""
Step 4 - generate all XAI saliency maps for every Cardiomegaly bbox image.

Loads each model ONCE, then loops images. Checkpointed: skips any (image,
method) whose .npy already exists. Verifies every saved map is (224,224).
Output: xai_maps/{model}/{img_id}_{method}.npy
"""
import os, time
import numpy as np
import pandas as pd
import torch
from xai_core import load_trained, preprocess, generate, METHODS

DATA_DIR  = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR = r"C:\Users\NMAMIT\cti_project\images"
XAI_DIR   = r"C:\Users\NMAMIT\cti_project\xai_maps"
MODELS    = ["densenet121", "resnet50", "efficientnet_b4"]


def bbox_images():
    bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    bb = bb[bb["Finding Label"] == "Cardiomegaly"]
    avail = set(os.listdir(IMAGE_DIR))
    return sorted(bb[bb["Image Index"].isin(avail)]["Image Index"].unique())


def main():
    imgs = bbox_images()
    print(f"Cardiomegaly bbox images on disk: {len(imgs)}")

    for name in MODELS:
        save_dir = os.path.join(XAI_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        model, target, fine = load_trained(name)
        t0 = time.time(); made = 0; skipped = 0
        for k, img in enumerate(imgs, 1):
            img_id = img.replace(".png", "")
            missing = [m for m in METHODS
                       if not os.path.exists(os.path.join(save_dir, f"{img_id}_{m}.npy"))]
            if not missing:
                skipped += 1
                continue
            x = preprocess(img)
            maps = generate(model, target, fine, x, methods=missing)
            for m, sal in maps.items():
                assert sal.shape == (224, 224), f"{name}/{img_id}_{m} shape {sal.shape}"
                np.save(os.path.join(save_dir, f"{img_id}_{m}.npy"), sal)
            made += 1
            if k % 20 == 0:
                print(f"  [{name}] {k}/{len(imgs)}  ({(time.time()-t0)/k:.1f}s/img)")
        del model
        torch.cuda.empty_cache()
        print(f"[{name}] done: {made} generated, {skipped} already complete, "
              f"{time.time()-t0:.0f}s")

    # ── Final verification ──
    print("\nVerification (files per method per model):")
    for name in MODELS:
        d = os.path.join(XAI_DIR, name)
        files = os.listdir(d)
        counts = {m: sum(f.endswith(f"_{m}.npy") for f in files) for m in METHODS}
        bad = [f for f in files if f.endswith(".npy") and np.load(os.path.join(d, f)).shape != (224, 224)]
        print(f"  {name}: " + " ".join(f"{m}={counts[m]}" for m in METHODS) +
              (f"  BAD_SHAPE={len(bad)}" if bad else "  shapes OK"))


if __name__ == "__main__":
    main()
