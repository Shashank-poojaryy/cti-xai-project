"""
Step 7 - publication figures (300 DPI, png + pdf). Figure 7 (ROC/PR) is made
by evaluate.py; this builds Figures 1-6.

  Fig 1 cti_heatmap          5 methods x 3 models, CTI annotated, best starred
  Fig 2 component_breakdown  grouped bars, 5 components/method, std error bars
  Fig 3 training_curves      loss + AUC, 3 models, best epoch marked
  Fig 4 sample_heatmaps      highest- vs lowest-CTI image, original + 5 overlays
  Fig 5 sensitivity_analysis grouped bars, 5 methods x 3 configs, ranks annotated
  Fig 6 pathology_correlation scatter bbox-area vs CTI per method + regression
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import spearmanr

DATA_DIR    = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR   = r"C:\Users\NMAMIT\cti_project\images"
XAI_DIR     = r"C:\Users\NMAMIT\cti_project\xai_maps"
MODELS_DIR  = r"C:\Users\NMAMIT\cti_project\models"
RESULTS_DIR = r"C:\Users\NMAMIT\cti_project\results"
MODELS  = ["densenet121", "resnet50", "efficientnet_b4"]
METHODS = ["gradcam", "gradcampp", "scorecam", "layercam", "integrated_gradients"]
MLABEL  = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "scorecam": "Score-CAM",
           "layercam": "Layer-CAM", "integrated_gradients": "Integrated Grad."}
ALABEL  = {"densenet121": "DenseNet121", "resnet50": "ResNet50",
           "efficientnet_b4": "EfficientNet-B4"}
COMPONENTS = ["localization", "stability", "cross_xai", "cross_arch", "sanity"]
CLABEL  = {"localization": "Localization", "stability": "Stability",
           "cross_xai": "Cross-XAI", "cross_arch": "Cross-Arch", "sanity": "Sanity"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png/.pdf")


def fig1_heatmap(perimg):
    piv = perimg.groupby(["method", "model"])["cti"].mean().reset_index() \
                .pivot(index="method", columns="model", values="cti").reindex(METHODS)[MODELS]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="CTI Score", fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([ALABEL[m] for m in MODELS])
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([MLABEL[m] for m in METHODS])
    best = np.unravel_index(np.nanargmax(piv.values), piv.values.shape)
    for i in range(len(METHODS)):
        for j in range(len(MODELS)):
            v = piv.values[i, j]
            txt = f"{v:.3f}" + ("★" if (i, j) == best else "")
            ax.text(j, i, txt, ha="center", va="center", fontweight="bold")
    ax.set_title("Composite Trustworthiness Index (CTI)\n5 XAI Methods x 3 Models", fontsize=13)
    save(fig, "cti_heatmap")


def fig2_components(perimg):
    means = perimg.groupby("method")[COMPONENTS].mean().reindex(METHODS)
    stds  = perimg.groupby("method")[COMPONENTS].std().reindex(METHODS)
    x = np.arange(len(METHODS)); w = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))
    for k, c in enumerate(COMPONENTS):
        ax.bar(x + (k - 2) * w, means[c], w, yerr=stds[c], capsize=3, label=CLABEL[c])
    ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in METHODS])
    ax.set_ylabel("Component Score"); ax.set_ylim(0, 1.05)
    ax.set_title("CTI Component Breakdown by XAI Method (mean ± std across images)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "component_breakdown")


def fig3_training():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for name in MODELS:
        f = os.path.join(MODELS_DIR, f"{name}_history.csv")
        if not os.path.exists(f):
            continue
        h = pd.read_csv(f)
        a1.plot(h["epoch"], h["train_loss"], "--", alpha=.6)
        a1.plot(h["epoch"], h["val_loss"], "-", label=ALABEL[name])
        a2.plot(h["epoch"], h["train_auc"], "--", alpha=.6)
        a2.plot(h["epoch"], h["val_auc"], "-", label=ALABEL[name])
        best = h.loc[h["val_auc"].idxmax(), "epoch"]
        a2.axvline(best, ls=":", color="gray", alpha=.5)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Training / Val Loss"); a1.legend()
    a2.set_xlabel("Epoch"); a2.set_ylabel("AUC-ROC"); a2.set_title("Training / Val AUC-ROC"); a2.legend()
    fig.suptitle("Training Curves (solid=val, dashed=train)", fontsize=13)
    save(fig, "training_curves")


def _overlay(ax, img_id, method, model, title):
    p = os.path.join(XAI_DIR, model, f"{img_id}_{method}.npy")
    img = Image.open(os.path.join(IMAGE_DIR, img_id + ".png")).convert("L").resize((224, 224))
    ax.imshow(img, cmap="gray")
    if os.path.exists(p):
        ax.imshow(np.load(p), cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=10); ax.axis("off")


def fig4_samples(perimg):
    best_model = perimg.groupby("model")["cti"].mean().idxmax()
    by_img = perimg[perimg["model"] == best_model].groupby("image_id")["cti"].mean()
    hi, lo = by_img.idxmax(), by_img.idxmin()
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    for r, (img_id, tag) in enumerate([(hi, "Highest CTI"), (lo, "Lowest CTI")]):
        orig = Image.open(os.path.join(IMAGE_DIR, img_id + ".png")).convert("L").resize((224, 224))
        axes[r, 0].imshow(orig, cmap="gray")
        axes[r, 0].set_title(f"{tag}\n{img_id}", fontsize=10); axes[r, 0].axis("off")
        for c, m in enumerate(METHODS, 1):
            row = perimg[(perimg.image_id == img_id) & (perimg.model == best_model) & (perimg.method == m)]
            cti = row["cti"].values[0] if len(row) else float("nan")
            _overlay(axes[r, c], img_id, m, best_model, f"{MLABEL[m]}\nCTI={cti:.3f}")
    fig.suptitle(f"Saliency Overlays ({ALABEL[best_model]})", fontsize=14)
    save(fig, "sample_heatmaps")


def fig5_sensitivity():
    f = os.path.join(RESULTS_DIR, "sensitivity_analysis.csv")
    if not os.path.exists(f):
        print("  (skip fig5: sensitivity_analysis.csv missing)"); return
    s = pd.read_csv(f)
    configs = list(s["config"].unique())
    x = np.arange(len(METHODS)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    for k, cfg in enumerate(configs):
        sub = s[s["config"] == cfg].set_index("method").reindex(METHODS)
        bars = ax.bar(x + (k - 1) * w, sub["cti"], w, label=cfg)
        for b, rk in zip(bars, sub["rank"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + .01, f"#{int(rk)}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in METHODS])
    ax.set_ylabel("CTI Score"); ax.set_title("CTI Sensitivity to Weight Configuration (rank annotated)")
    ax.legend(title="weights"); save(fig, "sensitivity_analysis")


def fig6_pathology(perimg):
    bb = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    bb = bb[bb["Finding Label"] == "Cardiomegaly"].drop_duplicates("Image Index")
    s2 = (224 / 1024) ** 2
    bb["area"] = (bb["w"] * bb["h]"]).astype(float) * s2
    bb["image_id"] = bb["Image Index"].str.replace(".png", "", regex=False)
    area = bb.set_index("image_id")["area"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, m in zip(axes, METHODS):
        sub = perimg[perimg["method"] == m].groupby("image_id")["cti"].mean()
        common = sub.index.intersection(area.index)
        a, c = area.loc[common].values, sub.loc[common].values
        ax.scatter(a, c, s=14, alpha=.5)
        if len(a) > 2:
            b1, b0 = np.polyfit(a, c, 1)
            xs = np.linspace(a.min(), a.max(), 50)
            ax.plot(xs, b0 + b1 * xs, "r-", lw=2)
            r, p = spearmanr(a, c)
            ax.set_title(f"{MLABEL[m]}\nr={r:.2f}, p={p:.3f}", fontsize=11)
        ax.set_xlabel("bbox area (px$^2$, 224)"); ax.set_ylabel("CTI")
    fig.suptitle("Pathology Region Size vs CTI", fontsize=14)
    save(fig, "pathology_correlation")


def main():
    perimg = pd.read_csv(os.path.join(RESULTS_DIR, "cti_per_image.csv"))
    print("Generating figures...")
    fig1_heatmap(perimg)
    fig2_components(perimg)
    fig3_training()
    fig4_samples(perimg)
    fig5_sensitivity()
    fig6_pathology(perimg)
    print("Figures 1-6 done (Figure 7 ROC/PR from evaluate.py).")


if __name__ == "__main__":
    main()
