"""
Streamlit app - XAI Reliability Assessment (CTI) for Cardiomegaly detection.

Multi-tab UI:
  - Analyze        : upload OR pick a demo image -> prediction gauge, GT-box
                     overlay, saliency overlay, live localization metrics,
                     CTI radar + trustworthiness rating, downloadable heatmap.
  - Compare Methods: all 5 saliency maps for one image/model, side by side.
  - Results        : project-level classification + CTI tables and figures.
  - About          : what CTI measures.

All values are read live from results/ (no hardcoded scores); saliency is made
with the fast xai_core engine (model cached once per session).
"""
import os, io
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import streamlit as st
from torchvision import transforms
from scipy.spatial.distance import jensenshannon
import xai_core

ROOT = r"C:\Users\NMAMIT\cti_project"
DATA_DIR, RESULTS, DEMO_DIR = (os.path.join(ROOT, d) for d in ("data", "results", "demo_images"))
IMAGE_DIR = os.path.join(ROOT, "images")  # full NIH set (gitignored); test images live here
DEVICE = xai_core.DEVICE

MODEL_LABEL = {"densenet121": "DenseNet121", "resnet50": "ResNet50", "efficientnet_b4": "EfficientNet-B4"}
METHOD_LABEL = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "scorecam": "Score-CAM",
                "layercam": "Layer-CAM", "integrated_gradients": "Integrated Gradients"}
COMPONENTS = ["localization", "stability", "cross_xai", "cross_arch", "sanity"]
COMP_LABEL = {"localization": "Localization", "stability": "Stability", "cross_xai": "Cross-XAI",
              "cross_arch": "Cross-Arch", "sanity": "Sanity"}
# Decision-threshold presets (keys match operating_points.csv "operating_point" column).
# The model is sensitivity-tuned (pos_weight + oversampling), so its raw scores sit high;
# 0.5 over-calls Cardiomegaly. The F1-optimal point restores ~94-97% specificity on normals.
OP_LABEL = {"F1_optimal": "Balanced (F1-optimal) — recommended",
            "Youden_J": "Youden's J (balanced sens/spec)",
            "high_sensitivity_recall>=0.90": "High sensitivity (recall ≥ 0.90)",
            "default_0.5": "Naïve 0.5 (over-calls positive)"}
_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

st.set_page_config(page_title="XAI Reliability (CTI)", layout="wide", page_icon="🫀")
st.markdown("""<style>
.block-container{padding-top:1.5rem;}
div[data-testid="stMetricValue"]{font-size:1.35rem;}
h1{color:#1f3b57;}
.stTabs [data-baseweb="tab"]{font-size:1.05rem;font-weight:600;}
</style>""", unsafe_allow_html=True)


# ─── cached resources ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(name):
    return xai_core.load_trained(name)


@st.cache_data(show_spinner=False)
def load_results():
    f = lambda n: pd.read_csv(os.path.join(RESULTS, n)) if os.path.exists(os.path.join(RESULTS, n)) else None
    return (f("cti_by_model_method.csv"), f("cti_results.csv"), f("cti_confidence_intervals.csv"),
            f("classification_performance.csv"), f("cti_per_image.csv"))


@st.cache_data(show_spinner=False)
def load_bbox():
    p = os.path.join(DATA_DIR, "BBox_Cardiomegaly.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).drop_duplicates("Image Index").set_index("Image Index")


@st.cache_data(show_spinner=False)
def load_test_index():
    """All 12,903 held-out test images with true labels (read live from test.csv)."""
    p = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)[["Image Index", "label"]].copy()
    df["true"] = df["label"].map({0: "Normal", 1: "Cardiomegaly"})
    return df


@st.cache_data(show_spinner=False)
def load_operating_points():
    p = os.path.join(RESULTS, "operating_points.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def get_threshold(op_df, model_name, op_key):
    """Per-model decision threshold from operating_points.csv (falls back to 0.5)."""
    if op_df is None:
        return 0.5
    r = op_df[(op_df["Model"] == MODEL_LABEL[model_name]) & (op_df["operating_point"] == op_key)]
    return float(r.iloc[0]["threshold"]) if not r.empty else 0.5


@st.cache_data(show_spinner=False)
def gen_maps(img_bytes, model_name, methods):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE); x.requires_grad = True
    model, target, fine = get_model(model_name)
    return xai_core.generate(model, target, fine, x, methods=methods)


@st.cache_data(show_spinner=False)
def predict(img_bytes, model_name):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    model, _, _ = get_model(model_name)
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).item())


# ─── helpers ───────────────────────────────────────────────────────────────
def overlay(img, sal, alpha=0.45):
    base = np.array(img.convert("RGB").resize((224, 224)))
    heat = (cm.jet(sal)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(((1 - alpha) * base + alpha * heat).astype(np.uint8))


def draw_box(img, row):
    im = img.convert("RGB").resize((224, 224)).copy(); s = 224 / 1024
    x, y, w, h = row["Bbox [x"] * s, row["y"] * s, row["w"] * s, row["h]"] * s
    ImageDraw.Draw(im).rectangle([x, y, x + w, y + h], outline=(0, 230, 0), width=3)
    return im


def localization(sal, row, n=224, orig=1024):
    s = n / orig
    x1, y1 = max(0, int(row["Bbox [x"] * s)), max(0, int(row["y"] * s))
    x2, y2 = min(n, int((row["Bbox [x"] + row["w"]) * s)), min(n, int((row["y"] + row["h]"]) * s))
    gt = np.zeros((n, n)); gt[y1:y2, x1:x2] = 1
    pred = (sal >= np.percentile(sal, 80)).astype(float)
    iou = (pred * gt).sum() / (((pred + gt) >= 1).sum() + 1e-8)
    sf = sal.flatten() + 1e-8; gf = gt.flatten() + 1e-8
    js = 1 - jensenshannon(sf / sf.sum(), gf / gf.sum())
    py, px = np.unravel_index(np.argmax(sal), sal.shape)
    pg = 1.0 if (y1 <= py <= y2 and x1 <= px <= x2) else 0.0
    return iou, js, pg


def prob_gauge(prob, thr=0.5):
    fig, ax = plt.subplots(figsize=(5, 0.9))
    ax.barh([0], [1], color="#e8e8e8", height=0.5)
    ax.barh([0], [prob], color=("#c0392b" if prob >= thr else "#27ae60"), height=0.5)
    ax.axvline(thr, color="#333", ls="--", lw=1)
    ax.text(prob, 0, f" {prob*100:.1f}%", va="center",
            ha="left" if prob < 0.85 else "right", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis("off")
    ax.text(thr, -0.55, f"threshold {thr:.3f}", ha="center", fontsize=7, color="#666")
    fig.tight_layout(); return fig


def radar(values):
    labels = [COMP_LABEL[c] for c in COMPONENTS]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    vals = list(values) + [values[0]]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.plot(ang, vals, color="#2e6da4", lw=2); ax.fill(ang, vals, color="#2e6da4", alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
    fig.tight_layout(); return fig


def trust_badge(cti):
    if cti >= 0.62: st.success(f"✅ HIGH TRUSTWORTHINESS — CTI {cti:.3f}")
    elif cti >= 0.55: st.warning(f"⚠️ MODERATE TRUSTWORTHINESS — CTI {cti:.3f}")
    else: st.error(f"❌ LOWER TRUSTWORTHINESS — CTI {cti:.3f}")


def demo_files():
    out = {}
    for sub in ("cardiomegaly", "normal"):
        d = os.path.join(DEMO_DIR, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"): out[f"{sub}/{f}"] = os.path.join(d, f)
    return out


def get_image_bytes(src_mode, uploaded, demo_choice, demos, test_choice=None):
    if src_mode == "Demo image" and demo_choice:
        with open(demos[demo_choice], "rb") as fh: return fh.read(), os.path.basename(demos[demo_choice])
    if src_mode == "Test set" and test_choice:
        p = os.path.join(IMAGE_DIR, test_choice)
        if os.path.exists(p):
            with open(p, "rb") as fh: return fh.read(), test_choice
    if src_mode == "Upload" and uploaded:
        return uploaded.getvalue(), uploaded.name
    return None, None


bymm, rank, ci, clf, per_img = load_results()
bbox = load_bbox()
op_df = load_operating_points()
test_idx = load_test_index()
demos = demo_files()
if per_img is not None:
    per_idx = per_img.set_index(["image_id", "model", "method"])
else:
    per_idx = None

st.title("🫀 XAI Reliability Assessment for Cardiomegaly Detection")
st.caption("Composite Trustworthiness Index (CTI) · NIH ChestX-ray14 · patient-independent split · "
           "best: Grad-CAM++ + DenseNet121 (CTI 0.715)")

tab_analyze, tab_compare, tab_results, tab_about = st.tabs(
    ["🔍 Analyze", "🆚 Compare Methods", "📊 Results", "ℹ️ About"])

# ════════════════════════ TAB 1 — ANALYZE ════════════════════════
with tab_analyze:
    c_in, c_out = st.columns([1, 2.4])
    with c_in:
        st.subheader("Input")
        has_testset = test_idx is not None and os.path.isdir(IMAGE_DIR)
        sources = ["Demo image"] + (["Test set"] if has_testset else []) + ["Upload"]
        src = st.radio("Image source", sources, horizontal=True)
        uploaded = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"]) if src == "Upload" else None
        demo_choice = st.selectbox("Pick a demo image", list(demos), index=0) if (src == "Demo image" and demos) else None
        test_choice = None
        if src == "Test set" and has_testset:
            cls = st.selectbox("Class filter", ["All", "Normal", "Cardiomegaly"])
            pool = test_idx if cls == "All" else test_idx[test_idx["true"] == cls]
            ids = pool["Image Index"].tolist()
            st.caption(f"{len(ids):,} held-out test images available (true label shown after Analyze).")
            test_choice = st.selectbox("Pick a test image", ids, index=0) if ids else None
        model_name = st.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get)
        method = st.selectbox("XAI method", list(METHOD_LABEL), format_func=METHOD_LABEL.get)
        op_key = st.selectbox("Decision threshold", list(OP_LABEL), format_func=OP_LABEL.get,
                              help="Imbalance-corrected models score high; 0.5 over-calls Cardiomegaly. "
                                   "F1-optimal restores ~94-97% specificity on normal X-rays.")
        go = st.button("Analyze", type="primary", use_container_width=True)

    with c_out:
        img_bytes, fname = get_image_bytes(src, uploaded, demo_choice, demos, test_choice)
        if not go:
            st.info("Choose an image, model, and XAI method, then click **Analyze**.")
        elif img_bytes is None:
            st.warning("Please select or upload an image first.")
        else:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            with st.spinner(f"{METHOD_LABEL[method]} on {MODEL_LABEL[model_name]}…"):
                prob = predict(img_bytes, model_name)
                sal = gen_maps(img_bytes, model_name, (method,))[method]
            thr = get_threshold(op_df, model_name, op_key)
            pred = "Cardiomegaly" if prob >= thr else "Normal"
            m1, m2, m3 = st.columns([1, 1, 1])
            m1.metric("Prediction", pred)
            m2.metric("Raw score", f"{prob*100:.1f}%", help=f"Decision threshold for this model: {thr:.3f}")
            m3.metric("Model", MODEL_LABEL[model_name])
            st.pyplot(prob_gauge(prob, thr))
            st.caption(f"ℹ️ {MODEL_LABEL[model_name]} is sensitivity-tuned, so raw scores sit high. "
                       f"Decision uses the **{OP_LABEL[op_key].split(' —')[0]}** threshold ({thr:.3f}); "
                       f"a score below it reads **Normal**. Always weigh the heatmap, not just the label.")
            if test_idx is not None:
                gt = test_idx[test_idx["Image Index"] == fname]
                if not gt.empty:
                    truth = gt.iloc[0]["true"]
                    ok = (truth == pred)
                    (st.success if ok else st.error)(
                        f"{'✅ Correct' if ok else '❌ Incorrect'} — ground truth: **{truth}** "
                        f"(held-out test set){'' if ok else f', predicted {pred}'}")

            has_box = bbox is not None and fname in bbox.index
            i1, i2 = st.columns(2)
            i1.image(draw_box(image, bbox.loc[fname]) if has_box else image.resize((224, 224)),
                     caption="Original" + (" + ground-truth box" if has_box else ""), use_container_width=True)
            i2.image(overlay(image, sal), caption=f"{METHOD_LABEL[method]} saliency", use_container_width=True)

            buf = io.BytesIO(); overlay(image, sal).save(buf, format="PNG")
            st.download_button("⬇️ Download heatmap", buf.getvalue(),
                               file_name=f"{os.path.splitext(fname)[0]}_{method}.png", mime="image/png")

            if has_box:
                iou, js, pg = localization(sal, bbox.loc[fname])
                st.markdown("**Live localization vs. ground-truth box**")
                l1, l2, l3 = st.columns(3)
                l1.metric("IoU", f"{iou:.3f}"); l2.metric("JS similarity", f"{js:.3f}")
                l3.metric("Pointing game", "hit" if pg else "miss")

            st.divider()
            img_id = os.path.splitext(fname)[0]
            row = None
            if per_idx is not None and (img_id, model_name, method) in per_idx.index:
                row = per_idx.loc[(img_id, model_name, method)]
                src_note = "per-image CTI (this exact image)"
            elif bymm is not None:
                r = bymm[(bymm.model == model_name) & (bymm.method == method)]
                if not r.empty: row = r.iloc[0]; src_note = "population-mean CTI for this model+method"
            if row is not None:
                st.subheader(f"CTI — {METHOD_LABEL[method]} on {MODEL_LABEL[model_name]}")
                st.caption(src_note)
                cc1, cc2 = st.columns([1.1, 1])
                with cc1:
                    cols = st.columns(3)
                    for k, c in enumerate(COMPONENTS):
                        cols[k % 3].metric(COMP_LABEL[c], f"{row[c]:.3f}")
                    st.metric("CTI", f"{row['cti']:.3f}"); trust_badge(float(row["cti"]))
                with cc2:
                    st.pyplot(radar([float(row[c]) for c in COMPONENTS]))

# ════════════════════════ TAB 2 — COMPARE METHODS ════════════════════════
with tab_compare:
    st.subheader("Compare all 5 XAI methods on one image")
    cc1, cc2 = st.columns([1, 1])
    csrc = cc1.radio("Image source", ["Demo image", "Upload"], horizontal=True, key="csrc")
    cup = cc1.file_uploader("Upload", type=["png", "jpg", "jpeg"], key="cup") if csrc == "Upload" else None
    cdemo = cc1.selectbox("Demo image", list(demos), key="cdemo") if (csrc == "Demo image" and demos) else None
    cmodel = cc2.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get, key="cmodel")
    cgo = cc2.button("Compare", type="primary")
    if cgo:
        cb, cfn = get_image_bytes(csrc, cup, cdemo, demos)
        if cb is None:
            st.warning("Select or upload an image first.")
        else:
            cimg = Image.open(io.BytesIO(cb)).convert("RGB")
            with st.spinner("Generating all 5 saliency maps…"):
                maps = gen_maps(cb, cmodel, tuple(METHOD_LABEL))
            cid = os.path.splitext(cfn)[0]
            # per-image CTI if known, else model-mean
            def mcti(m):
                if per_idx is not None and (cid, cmodel, m) in per_idx.index:
                    return float(per_idx.loc[(cid, cmodel, m)]["cti"])
                if bymm is not None:
                    r = bymm[(bymm.model == cmodel) & (bymm.method == m)]
                    return float(r.iloc[0]["cti"]) if not r.empty else float("nan")
                return float("nan")
            scores = {m: mcti(m) for m in METHOD_LABEL}
            best = max(scores, key=lambda k: (scores[k] if not np.isnan(scores[k]) else -1))
            cols = st.columns(6)
            cols[0].image(cimg.resize((224, 224)), caption="Original", use_container_width=True)
            for col, m in zip(cols[1:], METHOD_LABEL):
                star = " ⭐" if m == best else ""
                col.image(overlay(cimg, maps[m]),
                          caption=f"{METHOD_LABEL[m]}{star}\nCTI {scores[m]:.3f}", use_container_width=True)
            fig, ax = plt.subplots(figsize=(9, 2.6))
            ax.bar([METHOD_LABEL[m] for m in METHOD_LABEL], [scores[m] for m in METHOD_LABEL],
                   color=["#27ae60" if m == best else "#7fb3d5" for m in METHOD_LABEL])
            ax.set_ylim(0, 0.8); ax.set_ylabel("CTI"); ax.tick_params(axis="x", labelrotation=15)
            st.pyplot(fig)

# ════════════════════════ TAB 3 — RESULTS ════════════════════════
with tab_results:
    st.subheader("Classification performance (test set)")
    if clf is not None:
        st.dataframe(clf, use_container_width=True, hide_index=True)
    st.subheader("XAI trustworthiness ranking (mean CTI, 95% CI)")
    if ci is not None:
        show = ci.copy(); show["method"] = show["method"].map(METHOD_LABEL)
        show["95% CI"] = show.apply(lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)
        st.dataframe(show[["rank", "method", "mean_cti", "95% CI"]], use_container_width=True, hide_index=True)
    g1, g2 = st.columns(2)
    for col, fig in [(g1, "cti_heatmap.png"), (g2, "component_breakdown.png")]:
        p = os.path.join(RESULTS, fig)
        if os.path.exists(p): col.image(p, use_container_width=True)
    p = os.path.join(RESULTS, "roc_curves.png")
    if os.path.exists(p): st.image(p, use_container_width=True)

# ════════════════════════ TAB 4 — ABOUT ════════════════════════
with tab_about:
    st.subheader("What is the Composite Trustworthiness Index?")
    st.markdown("""
CTI measures *how much you can trust* a saliency explanation — not just whether the
prediction is right. It averages **five equally weighted dimensions** (each 0–1):

| Component | Question it answers |
|---|---|
| **Localization** | Does the heatmap land on the cardiac region (vs. the ground-truth box)? |
| **Stability** | Is the map robust to small input perturbations (noise, brightness, rotation)? |
| **Cross-XAI** | Do the five methods agree with each other? |
| **Cross-Arch** | Is the explanation consistent across DenseNet / ResNet / EfficientNet? |
| **Sanity** | Does the map depend on *learned* weights (vs. a randomized model)? |

**Dataset:** NIH ChestX-ray14, Cardiomegaly vs. Normal, strict patient-independent split
(zero patient leakage). **Models:** DenseNet121, ResNet50, EfficientNet-B4 (all AUC ≥ 0.83).
**Methods:** Grad-CAM, Grad-CAM++, Score-CAM, Layer-CAM, Integrated Gradients.

**Headline:** Grad-CAM++ is the most trustworthy method (mean CTI 0.662); best single
combination is **Grad-CAM++ + DenseNet121 (CTI 0.715)**.
""")
    st.caption("Tip: launch with the cti_project env — "
               "`...\\envs\\cti_project\\python.exe -m streamlit run app.py`")
