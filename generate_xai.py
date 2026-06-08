import os
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from captum.attr import GuidedGradCam, LayerGradCam, IntegratedGradients
from captum.attr import visualization as viz
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
DATA_DIR   = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR  = r'C:\Users\Acer\cti_project\data\images'
MODELS_DIR = r'C:\Users\Acer\cti_project\models'
XAI_DIR    = r'C:\Users\Acer\cti_project\xai_maps'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ─── TRANSFORMS ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── MODEL LOADER ─────────────────────────────────────────
def load_model(model_name):
    if model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        target_layer = model.features.denseblock4

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        target_layer = model.layer4

    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        target_layer = model.features[-1]

    weights_path = os.path.join(MODELS_DIR, f'{model_name}_best.pth')
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, target_layer


# ─── IMAGE LOADER ─────────────────────────────────────────
def load_image(img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True
    return tensor


# ─── NORMALIZE SALIENCY MAP ───────────────────────────────
def normalize_map(saliency):
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


# ─── GENERATE ALL 5 XAI MAPS ──────────────────────────────
def generate_xai_maps(model_name, img_name):
    model, target_layer = load_model(model_name)
    input_tensor = load_image(img_name)

    maps = {}

    # 1. Grad-CAM
    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode='bilinear', align_corners=False)
    maps['gradcam'] = normalize_map(
        attr.squeeze().detach().cpu().numpy())

    # 2. Grad-CAM++
    gradcampp = LayerGradCamPlusPlus(model, target_layer)
    attr = gradcampp.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode='bilinear', align_corners=False)
    maps['gradcampp'] = normalize_map(
        attr.squeeze().detach().cpu().numpy())

    # 3. Integrated Gradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)
    attr = ig.attribute(input_tensor, baseline, n_steps=50)
    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    maps['integrated_gradients'] = normalize_map(attr)

    # 4. Layer-CAM (uses an earlier layer for finer maps)
    if model_name == 'densenet121':
        fine_layer = model.features.denseblock3
    elif model_name == 'resnet50':
        fine_layer = model.layer3
    elif model_name == 'efficientnet_b4':
        fine_layer = model.features[-2]

    layercam = LayerGradCam(model, fine_layer)
    attr = layercam.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode='bilinear', align_corners=False)
    maps['layercam'] = normalize_map(
        attr.squeeze().detach().cpu().numpy())

    # 5. Score-CAM
    maps['scorecam'] = generate_scorecam(model, input_tensor, target_layer)

    return maps


# ─── SCORE-CAM (manual implementation) ───────────────────
def generate_scorecam(model, input_tensor, target_layer):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    act = activations[0].squeeze(0)  # shape: [C, H, W]
    scores = []

    with torch.no_grad():
        for i in range(act.shape[0]):
            upsampled = torch.nn.functional.interpolate(
                act[i].unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode='bilinear', align_corners=False)
            upsampled = (upsampled - upsampled.min()) / \
                        (upsampled.max() - upsampled.min() + 1e-8)
            masked = input_tensor * upsampled
            score = torch.sigmoid(model(masked)).item()
            scores.append(score)

    scores = torch.tensor(scores).to(DEVICE)
    weights = scores / (scores.sum() + 1e-8)

    scorecam = torch.zeros(act.shape[1], act.shape[2]).to(DEVICE)
    for i in range(act.shape[0]):
        scorecam += weights[i] * act[i]

    scorecam = scorecam.cpu().numpy()
    scorecam = np.maximum(scorecam, 0)
    return normalize_map(scorecam)


# ─── SAVE MAPS ────────────────────────────────────────────
def save_maps(maps, model_name, img_name):
    save_dir = os.path.join(XAI_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    img_id = img_name.replace('.png', '')
    for method, saliency in maps.items():
        save_path = os.path.join(save_dir, f'{img_id}_{method}.npy')
        np.save(save_path, saliency)


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("XAI generation script is ready.")
    print("Waiting for trained models and images.")
    print("Will generate 5 XAI maps per model-image pair.")
    print("Total combinations: 3 models x 5 methods = 15 maps per image.")