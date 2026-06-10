import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
from generate_xai import generate_xai_maps, load_model, normalize_map
from compute_sanity import compute_sanity
from compute_stability import compute_stability
import tempfile
import os

MODELS_DIR = r'C:\Users\Acer\cti_project\models'
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

with col2:
    if uploaded_file and analyze:
        with st.spinner("Running analysis..."):
            # Save uploaded file to temp
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Load image
            image = Image.open(temp_path).convert('RGB')

            # Load model and predict
            try:
                model, _ = load_model(model_name)
                img_tensor = transform(image).unsqueeze(0)
                prob = predict(model, img_tensor)
                prediction = "Cardiomegaly" if prob >= 0.5 else "Normal"
                confidence = prob if prob >= 0.5 else 1 - prob

                # Generate XAI map
                import shutil
                xai_image_dir = temp_dir
                original_dir = r'F:\cti_images\images'

                # Copy temp file to images dir for XAI generation
                dest_path = os.path.join(original_dir, uploaded_file.name)
                shutil.copy(temp_path, dest_path)

                maps = generate_xai_maps(model_name, uploaded_file.name)
                saliency = maps.get(xai_method)

                # Clean up
                if os.path.exists(dest_path):
                    os.remove(dest_path)

                # Display results
                st.subheader("Results")

                r1, r2, r3 = st.columns(3)
                r1.metric("Prediction", prediction)
                r2.metric("Confidence", f"{confidence*100:.1f}%")
                r3.metric("Model", model_name.upper())

                st.markdown("---")

                # Show images
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(image.resize((224, 224)), caption="Original X-ray", use_column_width=True)
                with img_col2:
                    if saliency is not None and saliency.shape == (224, 224):
                        overlay = overlay_heatmap(image, saliency)
                        st.image(overlay, caption=f"{xai_method} Heatmap", use_column_width=True)

                st.markdown("---")
                st.subheader("CTI Component Scores")

                # Quick CTI scores from saved results
                cti_data = {
                    'gradcam':    {'localization': 0.6722, 'stability': 0.9616, 'cross_xai': 0.5515, 'sanity': 0.7459, 'CTI': 0.7328},
                    'gradcampp':  {'localization': 0.6723, 'stability': 0.9605, 'cross_xai': 0.5515, 'sanity': 0.6894, 'CTI': 0.7184},
                    'layercam':   {'localization': 0.4757, 'stability': 0.8075, 'cross_xai': 0.5515, 'sanity': 0.7323, 'CTI': 0.6418},
                    'integrated_gradients': {'localization': 0.4832, 'stability': 0.6434, 'cross_xai': 0.5515, 'sanity': 0.9007, 'CTI': 0.6447},
                }

                scores = cti_data.get(xai_method, {})
                if scores:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Localization", f"{scores['localization']:.3f}")
                    c2.metric("Stability", f"{scores['stability']:.3f}")
                    c3.metric("Cross-XAI", f"{scores['cross_xai']:.3f}")
                    c4.metric("Sanity", f"{scores['sanity']:.3f}")
                    c5.metric("CTI Score", f"{scores['CTI']:.3f}", delta="Best" if xai_method == 'gradcam' else None)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the model weights are trained and saved in the models folder.")
    else:
        st.info("Upload a chest X-ray and click Analyze to begin.")
        st.markdown("""
        **About this tool:**
        - Evaluates XAI reliability using the Composite Trustworthiness Index (CTI)
        - Compares 4 saliency-based XAI methods
        - Tested on NIH ChestX-ray14 dataset
        
        **CTI Components:**
        - **Localization** — Does the heatmap focus on the cardiac region?
        - **Stability** — Is the heatmap consistent under small input changes?
        - **Cross-XAI Agreement** — Do all methods agree on the important region?
        - **Sanity Check** — Does the explanation depend on learned features?
        """)