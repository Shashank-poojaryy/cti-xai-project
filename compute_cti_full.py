"""
Step 5 - full CTI computation for every (image, model, method).

Optimized: each model (trained + its randomized twin) is loaded ONCE. Per image
the shared work (trained maps, 3 perturbations x 5 methods, random-model 5 maps)
is generated once, then the 5 components are read off per method.

Five components (each in [0,1], higher = more trustworthy), CTI = mean of the 5:
  1 Localization  (IoU + (1-JS) + PointingGame)/3   vs bbox ground truth
  2 Stability     mean over perturbations of (SSIM+Spearman)/2
  3 Cross-XAI     mean pairwise Spearman of the 5 methods (per image+model)
  4 Cross-Arch    mean pairwise Spearman of the 3 models (per image+method)
  5 Sanity        1 - (SSIM + max(Spearman,0))/2   vs SAME method on random model

Checkpointed to results/cti_per_image.csv (resume skips done rows).
Aggregated to results/cti_results.csv (per method, with ranks).
"""
import os, itertools
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity as ssim
from xai_core import (load_trained, load_random, preprocess, generate, METHODS,
                      XAI_DIR, DEVICE)
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR   = r"C:\Users\NMAMIT\cti_project\images"
RESULTS_DIR = r"C:\Users\NMAMIT\cti_project\results"
MODELS      = ["densenet121", "resnet50", "efficientnet_b4"]
PER_IMAGE   = os.path.join(RESULTS_DIR, "cti_per_image.csv")
WEIGHTS     = [0.20, 0.20, 0.20, 0.20, 0.20]
RANDOM_SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── component helpers ─────────────────────────────────────────────────────
def localization(saliency, bbox, img_size=224, orig=1024):
    x, y, w, h = [float(v) for v in bbox]
    s = img_size / orig
    x1, y1 = max(0, int(x * s)), max(0, int(y * s))
    x2, y2 = min(img_size, int((x + w) * s)), min(img_size, int((y + h) * s))
    gt = np.zeros((img_size, img_size)); gt[y1:y2, x1:x2] = 1
    thr = np.percentile(saliency, 80)
    pred = (saliency >= thr).astype(float)
    inter = (pred * gt).sum(); union = ((pred + gt) >= 1).sum()
    iou = inter / (union + 1e-8)
    sf = saliency.flatten() + 1e-8; sf /= sf.sum()
    gf = gt.flatten() + 1e-8; gf /= gf.sum()
    js = 1 - jensenshannon(sf, gf)
    py, px = np.unravel_index(np.argmax(saliency), saliency.shape)
    pg = 1.0 if (y1 <= py <= y2 and x1 <= px <= x2) else 0.0
    return (iou + js + pg) / 3, iou, js, pg


def safe_spearman(a, b):
    r, _ = spearmanr(a.flatten(), b.flatten())
    return 0.0 if np.isnan(r) else r


def mean_pairwise(maps):
    vals = [safe_spearman(a, b) for a, b in itertools.combinations(maps, 2)]
    return float(np.mean(vals)) if vals else 0.0


def perturbations(x):
    """3 perturbations per spec: gaussian noise, brightness, rotation.
    Returns list of (name, leaf tensor with requires_grad)."""
    g = torch.clamp(x + torch.randn_like(x) * 0.05, -3, 3)
    b = torch.clamp(x * 1.10, -3, 3)
    r = TF.rotate(x, 5)
    out = []
    for name, t in [("gaussian", g), ("brightness", b), ("rotation", r)]:
        out.append((name, t.detach().clone().requires_grad_(True)))
    return out


def load_saved(model_name, img_id):
    maps = {}
    for m in METHODS:
        p = os.path.join(XAI_DIR, model_name, f"{img_id}_{m}.npy")
        if os.path.exists(p):
            a = np.load(p)
            if a.shape == (224, 224):
                maps[m] = a
    return maps


# ─── main ──────────────────────────────────────────────────────────────────
def bbox_table():
    bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    bb = bb[bb["Finding Label"] == "Cardiomegaly"].copy()
    avail = set(os.listdir(IMAGE_DIR))
    bb = bb[bb["Image Index"].isin(avail)]
    bb = bb.drop_duplicates("Image Index").set_index("Image Index")
    return bb


def main():
    bb = bbox_table()
    imgs = sorted(bb.index.tolist())
    print(f"Computing CTI for {len(imgs)} images x {len(MODELS)} models x {len(METHODS)} methods")

    done = set()
    rows = []
    if os.path.exists(PER_IMAGE):
        prev = pd.read_csv(PER_IMAGE)
        rows = prev.to_dict("records")
        done = {(r["model"], r["method"], r["image_id"]) for r in rows}
        print(f"Resuming: {len(done)} rows already computed")

    torch.manual_seed(RANDOM_SEED)
    for name in MODELS:
        model, target, fine = load_trained(name)
        rmodel, rtarget, rfine = load_random(name, seed=RANDOM_SEED)
        for k, img in enumerate(imgs, 1):
            img_id = img.replace(".png", "")
            if all((name, m, img_id) in done for m in METHODS):
                continue
            bbox = (bb.loc[img, "Bbox [x"], bb.loc[img, "y"],
                    bb.loc[img, "w"], bb.loc[img, "h]"])

            trained = load_saved(name, img_id)             # from Step 4
            if len(trained) < len(METHODS):                # fallback: regenerate
                x0 = preprocess(img)
                trained = generate(model, target, fine, x0)

            x = preprocess(img)
            cross_xai = mean_pairwise([trained[m] for m in METHODS])

            # perturbed maps (all methods, once per perturbation)
            perturbed = {pn: generate(model, target, fine, pt)
                         for pn, pt in perturbations(x)}
            # random-model maps (same methods)
            xr = preprocess(img)
            random_maps = generate(rmodel, rtarget, rfine, xr)

            for m in METHODS:
                if (name, m, img_id) in done:
                    continue
                tmap = trained[m]
                loc, iou, js, pg = localization(tmap, bbox)

                # stability
                sc = []
                for pn, pmaps in perturbed.items():
                    pm = pmaps[m]
                    s = ssim(tmap, pm, data_range=1.0)
                    r = safe_spearman(tmap, pm)
                    sc.append((s + r) / 2)
                stability = float(np.mean(sc))

                # sanity (same method, random model)
                rs = ssim(tmap, random_maps[m], data_range=1.0)
                rr = max(safe_spearman(tmap, random_maps[m]), 0.0)
                sanity = 1.0 - (rs + rr) / 2

                # cross-architecture (this method across 3 models, from disk)
                arch_maps = []
                for mn in MODELS:
                    p = os.path.join(XAI_DIR, mn, f"{img_id}_{m}.npy")
                    if os.path.exists(p):
                        arch_maps.append(np.load(p))
                cross_arch = mean_pairwise(arch_maps) if len(arch_maps) >= 2 else 0.0

                cti = (WEIGHTS[0]*loc + WEIGHTS[1]*stability + WEIGHTS[2]*cross_xai +
                       WEIGHTS[3]*cross_arch + WEIGHTS[4]*sanity)
                rows.append({"image_id": img_id, "model": name, "method": m,
                             "localization": round(loc, 4), "iou": round(iou, 4),
                             "js_div": round(js, 4), "pointing_game": pg,
                             "stability": round(stability, 4),
                             "cross_xai": round(cross_xai, 4),
                             "cross_arch": round(cross_arch, 4),
                             "sanity": round(sanity, 4), "cti": round(cti, 4)})
                done.add((name, m, img_id))

            # checkpoint after every image
            pd.DataFrame(rows).to_csv(PER_IMAGE, index=False)
            if k % 20 == 0:
                print(f"  [{name}] {k}/{len(imgs)} images")
        del model, rmodel
        torch.cuda.empty_cache()
        print(f"[{name}] complete")

    aggregate()


def aggregate():
    df = pd.read_csv(PER_IMAGE)
    # mean per model-method
    g = df.groupby(["model", "method"]).agg(
        localization=("localization", "mean"), stability=("stability", "mean"),
        cross_xai=("cross_xai", "mean"), cross_arch=("cross_arch", "mean"),
        sanity=("sanity", "mean"), cti=("cti", "mean")).reset_index().round(4)
    g.to_csv(os.path.join(RESULTS_DIR, "cti_by_model_method.csv"), index=False)

    # method-level pivot (mean across models) with ranks -> Module 5 cti_results.csv
    piv = df.groupby("method")["cti"].mean()
    out = pd.DataFrame({"method": piv.index})
    for mn in MODELS:
        mm = df[df["model"] == mn].groupby("method")["cti"].mean()
        out[f"{mn}_cti"] = out["method"].map(mm).round(4)
    out["mean_cti"] = out["method"].map(piv).round(4)
    out = out.sort_values("mean_cti", ascending=False).reset_index(drop=True)
    out["rank"] = out["mean_cti"].rank(ascending=False).astype(int)
    out.to_csv(os.path.join(RESULTS_DIR, "cti_results.csv"), index=False)
    print("\nCTI results (method ranking):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
