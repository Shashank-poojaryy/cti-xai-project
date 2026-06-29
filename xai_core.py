"""
Shared XAI engine. Loads a model ONCE and generates saliency maps for any
image/method, so callers never reload weights per image (the old code reloaded
all 3 models for every image -> the ~2-day runtime). Used by both XAI map
generation (Step 4) and CTI computation (Step 5: stability / sanity).

Methods (all normalized to [0,1], all returned at 224x224):
  gradcam, gradcampp, scorecam, layercam, integrated_gradients

Math is kept identical to the validated generate_xai.py (incl. the Score-CAM
ReLU-before-combine fix for non-post-ReLU target layers).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from captum.attr import LayerGradCam, IntegratedGradients
import warnings
warnings.filterwarnings("ignore")

IMAGE_DIR  = r"C:\Users\NMAMIT\cti_project\images"
MODELS_DIR = r"C:\Users\NMAMIT\cti_project\models"
XAI_DIR    = r"C:\Users\NMAMIT\cti_project\xai_maps"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["gradcam", "gradcampp", "scorecam", "layercam", "integrated_gradients"]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def normalize_to_01(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def _build(name):
    """Build architecture + return (model, target_layer, fine_layer)."""
    if name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        return m, m.features.denseblock4, m.features.denseblock3
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m, m.layer4, m.layer3
    if name == "efficientnet_b4":
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return m, m.features[-1], m.features[-2]
    raise ValueError(name)


def load_trained(name):
    """Load a trained model once. Returns (model, target_layer, fine_layer)."""
    m, target, fine = _build(name)
    m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_best.pth"),
                                 map_location=DEVICE))
    m.eval().to(DEVICE)
    return m, target, fine


def load_random(name, seed=0):
    """Build a fully-randomized model (for the sanity check). Reproducible."""
    torch.manual_seed(seed)
    m, target, fine = _build(name)
    for layer in m.modules():
        if hasattr(layer, "weight") and layer.weight is not None:
            if layer.weight.dim() >= 2:
                nn.init.kaiming_uniform_(layer.weight)
            else:
                nn.init.uniform_(layer.weight, -0.1, 0.1)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    m.eval().to(DEVICE)
    return m, target, fine


def preprocess(img_name):
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    t = _transform(img).unsqueeze(0).to(DEVICE)
    t.requires_grad = True
    return t


def tensor_input(tensor):
    """For perturbed tensors already in model space."""
    t = tensor.clone().detach().to(DEVICE)
    t.requires_grad = True
    return t


# ─── METHODS (each takes an already-loaded model) ──────────────────────────
def m_gradcam(model, target_layer, x):
    gc = LayerGradCam(model, target_layer)
    attr = gc.attribute(x, target=None)
    attr = F.interpolate(attr, size=(224, 224), mode="bilinear", align_corners=False)
    return normalize_to_01(attr.squeeze().detach().cpu().numpy())


def m_layercam(model, fine_layer, x):
    # True Layer-CAM (Jiang et al. 2021): element-wise positive-gradient
    # weighting of a finer layer's activations (NOT global-average-pooled
    # gradients like Grad-CAM).  w_ij^k = ReLU(dY/dA_ij^k);
    # L = ReLU( sum_k w_ij^k * A_ij^k ).
    acts, grads = [], []
    h1 = fine_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = fine_layer.register_full_backward_hook(lambda m, i, o: grads.append(o[0].detach()))
    out = model(x)
    model.zero_grad()
    out.backward(retain_graph=False)
    h1.remove(); h2.remove()
    A = acts[0].detach()
    w = torch.relu(grads[0])
    cam = torch.relu((w * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    return normalize_to_01(cam.squeeze().detach().cpu().numpy())


def m_gradcampp(model, target_layer, x):
    acts, grads = [], []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, i, o: grads.append(o[0].detach()))
    out = model(x)
    model.zero_grad()
    out.backward(retain_graph=False)
    h1.remove(); h2.remove()
    act = acts[0].detach(); grad = grads[0]
    g2, g3 = grad ** 2, grad ** 3
    alpha = g2 / (2 * g2 + act * g3 + 1e-8)
    w = (alpha * torch.relu(grad)).mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    return normalize_to_01(cam.squeeze().detach().cpu().numpy())


SCORECAM_K = 64   # per spec Module 4: limit channels to min(64, n) for speed

def m_scorecam(model, target_layer, x):
    acts = []
    h = target_layer.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        _ = model(x)
    h.remove()
    act = acts[0].squeeze(0)                      # (C, h, w)
    act = torch.relu(act)                         # target layers are not post-ReLU
    C = act.shape[0]
    k = min(SCORECAM_K, C)
    # pick the top-k channels by mean activation
    topk = torch.topk(act.mean(dim=(1, 2)), k).indices
    sel = act[topk]                               # (k, h, w)
    # upsample + per-channel [0,1] normalize -> masks
    up = F.interpolate(sel.unsqueeze(1), size=(224, 224),
                       mode="bilinear", align_corners=False)        # (k,1,224,224)
    mn = up.amin(dim=(2, 3), keepdim=True); mx = up.amax(dim=(2, 3), keepdim=True)
    up = (up - mn) / (mx - mn + 1e-8)
    with torch.no_grad():
        masked = x * up                            # (k,3,224,224) broadcast
        scores = torch.sigmoid(model(masked)).squeeze(1)            # one batched forward
    weights = torch.softmax(scores, dim=0)         # per spec: softmax(scores)
    cam = (weights.view(k, 1, 1) * sel).sum(0)     # weighted sum of activation maps
    cam = torch.relu(cam)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0).float(), size=(224, 224),
                        mode="bilinear", align_corners=False)
    out = normalize_to_01(cam.squeeze().cpu().numpy())
    assert out.shape == (224, 224), f"Score-CAM bad shape {out.shape}"
    return out


def m_integrated_gradients(model, x):
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x).to(DEVICE)
    attr = ig.attribute(x, baseline, n_steps=50)
    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    return normalize_to_01(attr)


def generate(model, target_layer, fine_layer, x, methods=METHODS):
    """Generate the requested methods for a single preprocessed input x."""
    out = {}
    for mth in methods:
        if mth == "gradcam":
            out[mth] = m_gradcam(model, target_layer, x)
        elif mth == "gradcampp":
            out[mth] = m_gradcampp(model, target_layer, x)
        elif mth == "scorecam":
            out[mth] = m_scorecam(model, target_layer, x)
        elif mth == "layercam":
            out[mth] = m_layercam(model, fine_layer, x)
        elif mth == "integrated_gradients":
            out[mth] = m_integrated_gradients(model, x)
    return out
