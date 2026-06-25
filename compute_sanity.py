import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
from generate_xai import generate_xai_maps, load_model, normalize_map
from captum.attr import LayerGradCam
import warnings
warnings.filterwarnings('ignore')

IMAGE_DIR  = r'C:\Users\NMAMIT\cti_project\images'
MODELS_DIR = r'C:\Users\NMAMIT\cti_project\models'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def build_random_model(model_name):
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

    # Randomize all weights
    for layer in model.modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.dim() >= 2:
                nn.init.kaiming_uniform_(layer.weight)
            else:
                nn.init.uniform_(layer.weight, -0.1, 0.1)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)

    model.eval()
    model.to(DEVICE)
    return model, target_layer

def get_gradcam_map(model, target_layer, img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True

    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(tensor, target=None)
    attr = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode='bilinear', align_corners=False)
    return normalize_map(attr.squeeze().detach().cpu().numpy())

def compute_sanity(model_name, method, img_name):
    # Step 1 — trained model map
    trained_maps = generate_xai_maps(model_name, img_name)
    if method not in trained_maps:
        return None
    trained_map = trained_maps[method]
    if trained_map.shape != (224, 224):
        return None

    # Step 2 — random model map (always use gradcam for random — others need trained weights)
    random_model, target_layer = build_random_model(model_name)
    random_map = get_gradcam_map(random_model, target_layer, img_name)

    # Step 3 — compare: large difference = trustworthy
    s = ssim(trained_map, random_map, data_range=1.0)
    r, _ = spearmanr(trained_map.flatten(), random_map.flatten())
    r = max(r, 0) if not np.isnan(r) else 0

    # Invert: high difference = high sanity score
    if np.isnan(s) or np.isnan(r):
        return 0.5  # neutral score if undefined
    r = max(r, 0)
    sanity = 1.0 - ((s + r) / 2)
    return float(sanity)


if __name__ == '__main__':
    import pandas as pd

    DATA_DIR = r'C:\Users\NMAMIT\cti_project\data'

    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]

    test_imgs = matches['Image Index'].values[:2]

    for img_name in test_imgs:
        print(f'\nImage: {img_name}')
        for method in ['gradcam', 'gradcampp']:
            score = compute_sanity('densenet121', method, img_name)
            print(f'  {method} sanity: {score}')