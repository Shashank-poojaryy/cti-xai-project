import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.cm as cm
from generate_xai import generate_xai_maps, load_model, normalize_map
from compute_sanity import compute_sanity
from compute_stability import compute_stability
import tempfile
import os
import shutil

MODELS_DIR = r'C:\Users\NMAMIT\cti_project\models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
        prob = torch.sigmoid(output).item()
    return prob

def overlay_heatmap(original_img, saliency_map):
    img_array = np.array(original_img.resize((224, 224)))
    heatmap = cm.jet(saliency_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)

# ── CTI Data — Complete 5-component results ───────────────
cti_data = {
    'gradcam': {
        'localization': 0.6722, 'stability': 0.9578,
        'cross_xai': 0.5515, 'cross_arch': 0.2007,
        'sanity': 0.8375, 'CTI': 0.5745
    },
    'gradcampp': {
        'localization': 0.6723, 'stability': 0.9561,
        'cross_xai': 0.5515, 'cross_arch': 0.2515,
        'sanity': 0.6832, 'CTI': 0.5612
    },
    'layercam': {
        'localization': 0.4757, 'stability': 0.8060,
        'cross_xai': 0.5515, 'cross_arch': 0.0294,
        'sanity': 0.6995, 'CTI': 0.4499
    },
    'integrated_gradients': {
        'localization': 0.4832, 'stability': 0.6358,
        'cross_xai': 0.5515, 'cross_arch': 0.4432,
        'sanity': 0.9403, 'CTI': 0.5362
    },
}

# ── Streamlit UI ───────────────────────────────────────────
st.set_page_config(page_title="XAI Reliability Assessment", layout="wide")

st.title("🫀 XAI Reliability Assessment for Cardiomegaly Detection")
st.markdown("**Composite Trustworthiness Index (CTI) — Saliency-Based XAI Evaluation**")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Settings")
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
    model_name = st.selectbox("Select Model", ['densenet121', 'resnet50', 'efficientnet_b4'])
    xai_method = st.selectbox("Select XAI Method", ['gradcam', 'gradcampp', 'layercam', 'integrated_gradients'])
    analyze = st.button("Analyze", type="primary")

    st.markdown("---")
    st.markdown("**CTI Components:**")
    st.markdown("- **Localization** — Heatmap focuses on cardiac region?")
    st.markdown("- **Stability** — Consistent under small input changes?")
    st.markdown("- **Cross-XAI** — Do all methods agree?")
    st.markdown("- **Cross-Arch** — Consistent across architectures?")
    st.markdown("- **Sanity** — Depends on learned features?")

with col2:
    if uploaded_file and analyze:
        with st.spinner("Running analysis..."):
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open(temp_path).convert('RGB')

            try:
                model, _ = load_model(model_name)
                img_tensor = transform(image).unsqueeze(0)
                prob = predict(model, img_tensor)
                prediction = "Cardiomegaly" if prob >= 0.5 else "Normal"
                confidence = prob if prob >= 0.5 else 1 - prob

                original_dir = r'C:\Users\NMAMIT\cti_project\images'
                dest_path = os.path.join(original_dir, uploaded_file.name)
                shutil.copy(temp_path, dest_path)

                maps = generate_xai_maps(model_name, uploaded_file.name)
                saliency = maps.get(xai_method)

                if os.path.exists(dest_path):
                    os.remove(dest_path)

                # ── Results header ─────────────────────────
                st.subheader("Results")
                r1, r2, r3 = st.columns(3)
                r1.metric("Prediction", prediction)
                r2.metric("Confidence", f"{confidence*100:.1f}%")
                r3.metric("Model", model_name.upper())

                st.markdown("---")

                # ── Images ─────────────────────────────────
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(image.resize((224, 224)),
                             caption="Original X-ray",
                             use_container_width=True)
                with img_col2:
                    if saliency is not None and saliency.shape == (224, 224):
                        overlay = overlay_heatmap(image, saliency)
                        st.image(overlay,
                                 caption=f"{xai_method} Heatmap",
                                 use_container_width=True)

                st.markdown("---")

                # ── CTI Scores ─────────────────────────────
                st.subheader("CTI Component Scores")
                scores = cti_data.get(xai_method, {})
                if scores:
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Localization", f"{scores['localization']:.3f}")
                    c2.metric("Stability",    f"{scores['stability']:.3f}")
                    c3.metric("Cross-XAI",    f"{scores['cross_xai']:.3f}")
                    c4.metric("Cross-Arch",   f"{scores['cross_arch']:.3f}")
                    c5.metric("Sanity",       f"{scores['sanity']:.3f}")
                    c6.metric("CTI Score",    f"{scores['CTI']:.3f}",
                              delta="Best" if xai_method == 'gradcam' else None)

                    st.markdown("---")

                    # ── Trustworthiness Rating ──────────────
                    st.subheader("🏆 Trustworthiness Rating")
                    cti_val = scores['CTI']
                    if cti_val >= 0.55:
                        st.success(f"✅ HIGH TRUSTWORTHINESS — CTI: {cti_val:.3f} — This XAI method is clinically reliable for Cardiomegaly detection.")
                    elif cti_val >= 0.50:
                        st.warning(f"⚠️ MODERATE TRUSTWORTHINESS — CTI: {cti_val:.3f} — Use with caution in clinical settings.")
                    else:
                        st.error(f"❌ LOW TRUSTWORTHINESS — CTI: {cti_val:.3f} — This method is not recommended for clinical use.")

                    # ── Ranking ────────────────────────────
                    st.markdown("---")
                    st.subheader("📊 Method Ranking")
                    ranking_data = {
                        'Method': ['Grad-CAM', 'Grad-CAM++', 'Integrated Gradients', 'Layer-CAM'],
                        'CTI Score': [0.5745, 0.5612, 0.5362, 0.4499],
                        'Rank': ['🥇 1st', '🥈 2nd', '🥉 3rd', '4th']
                    }
                    import pandas as pd
                    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure model weights are trained and saved in the models folder.")
    else:
        st.info("Upload a chest X-ray and click Analyze to begin.")
        st.markdown("""
        **About this tool:**
        - Evaluates XAI reliability using the Composite Trustworthiness Index (CTI)
        - Compares 4 saliency-based XAI methods across 5 reliability dimensions
        - Tested on NIH ChestX-ray14 dataset
        - Novel contribution: first CTI framework for medical XAI evaluation
        
        **How to use:**
        1. Upload a chest X-ray image
        2. Select a model and XAI method
        3. Click Analyze
        4. View prediction, heatmap, and CTI trustworthiness rating
        """)