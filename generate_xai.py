import os
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from captum.attr import LayerGradCam, IntegratedGradients
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = r"C:\Users\NMAMIT\cti_project\data"
IMAGE_DIR  = r"C:\Users\NMAMIT\cti_project\images"
MODELS_DIR = r"C:\Users\NMAMIT\cti_project\models"
XAI_DIR    = r"C:\Users\NMAMIT\cti_project\xai_maps"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def load_model(model_name):
    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        target_layer = model.features.denseblock4
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        target_layer = model.layer4
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        target_layer = model.features[-1]
    weights_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, target_layer

def load_image(img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True
    return tensor

def normalize_map(saliency):
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency

def generate_gradcampp(model, input_tensor, target_layer):
    activations_pp, gradients_pp = [], []
    def forward_hook(m, i, o): activations_pp.append(o.detach())
    def backward_hook(m, i, o): gradients_pp.append(o[0].detach())
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    output = model(input_tensor)
    model.zero_grad()
    output.backward(retain_graph=True)
    h1.remove()
    h2.remove()
    act = activations_pp[0]
    grad = gradients_pp[0]
    grad2 = grad ** 2
    grad3 = grad ** 3
    alpha = grad2 / (2 * grad2 + act * grad3 + 1e-8)
    weights = (alpha * torch.relu(grad)).mean(dim=(2,3), keepdim=True)
    gcpp = torch.relu((weights * act).sum(dim=1, keepdim=True))
    gcpp = torch.nn.functional.interpolate(gcpp, size=(224,224), mode="bilinear", align_corners=False)
    return normalize_map(gcpp.squeeze().detach().cpu().numpy())

def generate_scorecam(model, input_tensor, target_layer):
    activations = []
    def hook_fn(module, input, output): activations.append(output.detach())
    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()
    act = activations[0].squeeze(0)
    scores = []
    with torch.no_grad():
        for i in range(act.shape[0]):
            upsampled = torch.nn.functional.interpolate(
                act[i].unsqueeze(0).unsqueeze(0), size=(224,224),
                mode="bilinear", align_corners=False)
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
            masked = input_tensor * upsampled
            score = torch.sigmoid(model(masked)).item()
            scores.append(score)
    scores = torch.tensor(scores).to(DEVICE)
    weights = scores / (scores.sum() + 1e-8)
    # ReLU activations before combining: target layers (e.g. DenseNet
    # denseblock4) are not post-ReLU, so raw mixed-sign activations make the
    # weighted sum non-positive everywhere -> an all-zero map. Canonical
    # Score-CAM assumes non-negative activation maps.
    act = torch.relu(act)
    scorecam = torch.zeros(act.shape[1], act.shape[2]).to(DEVICE)
    for i in range(act.shape[0]):
        scorecam += weights[i] * act[i]
    scorecam = scorecam.cpu().numpy()
    scorecam = np.maximum(scorecam, 0)
    import torch.nn.functional as F
    st = torch.tensor(scorecam).unsqueeze(0).unsqueeze(0).float()
    su = F.interpolate(st, size=(224,224), mode="bilinear", align_corners=False)
    return normalize_map(su.squeeze().numpy())

def generate_xai_maps(model_name, img_name):
    model, target_layer = load_model(model_name)
    input_tensor = load_image(img_name)
    maps = {}

    # 1. Grad-CAM
    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(attr, size=(224,224), mode="bilinear", align_corners=False)
    maps["gradcam"] = normalize_map(attr.squeeze().detach().cpu().numpy())

    # 2. Grad-CAM++
    maps["gradcampp"] = generate_gradcampp(model, input_tensor, target_layer)

    # 3. Integrated Gradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)
    attr = ig.attribute(input_tensor, baseline, n_steps=50)
    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    maps["integrated_gradients"] = normalize_map(attr)

    # 4. Layer-CAM
    if model_name == "densenet121":
        fine_layer = model.features.denseblock3
    elif model_name == "resnet50":
        fine_layer = model.layer3
    elif model_name == "efficientnet_b4":
        fine_layer = model.features[-2]
    layercam = LayerGradCam(model, fine_layer)
    attr = layercam.attribute(input_tensor, target=None)
    attr = torch.nn.functional.interpolate(attr, size=(224,224), mode="bilinear", align_corners=False)
    maps["layercam"] = normalize_map(attr.squeeze().detach().cpu().numpy())

    # 5. Score-CAM
    maps["scorecam"] = generate_scorecam(model, input_tensor, target_layer)

    return maps

def save_maps(maps, model_name, img_name):
    save_dir = os.path.join(XAI_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    img_id = img_name.replace(".png", "")
    for method, saliency in maps.items():
        save_path = os.path.join(save_dir, f"{img_id}_{method}.npy")
        np.save(save_path, saliency)

if __name__ == "__main__":
    import pandas as pd
    test_df = pd.read_csv(r"C:\Users\NMAMIT\cti_project\data\test.csv")
    cardio_imgs = test_df[test_df["label"]==1]["Image Index"].values[:3]
    for model_name in ["densenet121"]:
        print(f"Generating XAI maps for {model_name}")
        for img_name in cardio_imgs:
            print(f"  Processing {img_name}")
            maps = generate_xai_maps(model_name, img_name)
            save_maps(maps, model_name, img_name)
            print(f"  Saved {len(maps)} maps")
    print("Done.")
