import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from generate_xai import generate_xai_maps, save_maps, load_model
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
DATA_DIR   = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR  = r'F:\cti_images\images'
MODELS_DIR = r'C:\Users\Acer\cti_project\models'
XAI_DIR    = r'C:\Users\Acer\cti_project\xai_maps'
RESULTS_DIR= r'C:\Users\Acer\cti_project\results'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAMES = ['densenet121', 'resnet50', 'efficientnet_b4']
XAI_METHODS = ['gradcam', 'gradcampp', 'scorecam', 'layercam', 'integrated_gradients']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─── LOAD SALIENCY MAP ────────────────────────────────────
def load_map(model_name, img_id, method):
    path = os.path.join(XAI_DIR, model_name, f'{img_id}_{method}.npy')
    return np.load(path)


# ─── METRIC 1: LOCALIZATION ───────────────────────────────
def compute_localization(saliency, bbox, img_size=224):
    x, y, w, h = bbox
    scale_x = img_size / 1024
    scale_y = img_size / 1024
    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + w) * scale_x)
    y2 = int((y + h) * scale_y)

    # Binary mask from bounding box
    gt_mask = np.zeros((img_size, img_size))
    gt_mask[y1:y2, x1:x2] = 1

    # Threshold saliency at top 20%
    threshold = np.percentile(saliency, 80)
    pred_mask = (saliency >= threshold).astype(float)

    # IoU
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) >= 1).sum()
    iou = intersection / (union + 1e-8)

    # JS Divergence
    sal_flat = saliency.flatten() + 1e-8
    gt_flat  = gt_mask.flatten() + 1e-8
    sal_flat = sal_flat / sal_flat.sum()
    gt_flat  = gt_flat / gt_flat.sum()
    js_div = 1 - jensenshannon(sal_flat, gt_flat)  # invert so higher=better

    # Pointing Game
    peak_y, peak_x = np.unravel_index(np.argmax(saliency), saliency.shape)
    pointing = 1.0 if (y1 <= peak_y <= y2 and x1 <= peak_x <= x2) else 0.0

    localization = (iou + js_div + pointing) / 3
    return localization, iou, js_div, pointing


# ─── METRIC 2: STABILITY ──────────────────────────────────
def compute_stability(model_name, method, img_name, img_tensor):
    original_map = generate_xai_maps(model_name, img_name)[method]
    scores_ssim, scores_spearman = [], []

    perturbations = [
        ('gaussian', lambda x: x + torch.randn_like(x) * 0.05),
        ('brightness_up', lambda x: x * 1.10),
        ('brightness_down', lambda x: x * 0.90),
        ('rotation', lambda x: transforms.functional.rotate(x, 5)),
    ]

    model, _ = load_model(model_name)

    for name, perturb_fn in perturbations:
        perturbed = perturb_fn(img_tensor.clone())
        perturbed = torch.clamp(perturbed, -3, 3)

        temp_path = os.path.join(IMAGE_DIR, '_temp_perturbed.png')
        perturbed_pil = transforms.ToPILImage()(perturbed.squeeze().cpu())
        perturbed_pil.save(temp_path)

        perturbed_map = generate_xai_maps(model_name, '_temp_perturbed.png')[method]

        s = ssim(original_map, perturbed_map, data_range=1.0)
        r, _ = spearmanr(original_map.flatten(), perturbed_map.flatten())
        scores_ssim.append(s)
        scores_spearman.append(r)

    stability = (np.mean(scores_ssim) + np.mean(scores_spearman)) / 2
    return stability


# ─── METRIC 3: CROSS-XAI AGREEMENT ───────────────────────
def compute_cross_xai_agreement(model_name, img_id):
    maps = {m: load_map(model_name, img_id, m) for m in XAI_METHODS}
    pairs = []
    methods = list(maps.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            r, _ = spearmanr(maps[methods[i]].flatten(),
                             maps[methods[j]].flatten())
            pairs.append(r)
    return np.mean(pairs)


# ─── METRIC 4: CROSS-ARCHITECTURE AGREEMENT ──────────────
def compute_cross_arch_agreement(method, img_id):
    maps = {m: load_map(m, img_id, method) for m in MODEL_NAMES}
    pairs = []
    model_list = list(maps.keys())
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            r, _ = spearmanr(maps[model_list[i]].flatten(),
                             maps[model_list[j]].flatten())
            pairs.append(r)
    return np.mean(pairs)


# ─── METRIC 5: SANITY CHECK ───────────────────────────────
def compute_sanity_check(model_name, method, img_name):
    trained_map = generate_xai_maps(model_name, img_name)[method]

    # Load model and randomize weights
    model, target_layer = load_model(model_name)
    for layer in model.modules():
        if hasattr(layer, 'weight'):
            nn.init.normal_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    model.eval()

    # Generate random map using same image
    from generate_xai import normalize_map
    from captum.attr import LayerGradCam
    input_tensor = transform(
        Image.open(os.path.join(IMAGE_DIR, img_name)).convert('RGB')
    ).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode='bilinear', align_corners=False)
    random_map = normalize_map(attr.squeeze().detach().cpu().numpy())

    s = ssim(trained_map, random_map, data_range=1.0)
    r, _ = spearmanr(trained_map.flatten(), random_map.flatten())

    # Large difference = trustworthy
    sanity = 1 - ((s + max(r, 0)) / 2)
    return sanity


# ─── COMPUTE FULL CTI ─────────────────────────────────────
def compute_cti(localization, stability, cross_xai, cross_arch, sanity):
    return 0.20 * localization + \
           0.20 * stability    + \
           0.20 * cross_xai   + \
           0.20 * cross_arch  + \
           0.20 * sanity


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("CTI computation script is ready.")
    print("Components:")
    print("  1. Localization  (IoU + JS_DIV + Pointing Game)")
    print("  2. Stability     (SSIM + Spearman across perturbations)")
    print("  3. Cross-XAI Agreement     (10 pairwise Spearman)")
    print("  4. Cross-Architecture Agreement (3 pairwise Spearman)")
    print("  5. Sanity Check  (trained vs random weights)")
    print("\nCTI = 0.20 x each component")
    print("Total combinations: 3 models x 5 XAI methods = 15 CTI scores")