import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
from generate_xai import generate_xai_maps, load_model
import warnings
warnings.filterwarnings('ignore')

IMAGE_DIR  = r'C:\Users\NMAMIT\cti_project\images'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_image_tensor(img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

def perturb(tensor, mode):
    t = tensor.clone()
    if mode == 'gaussian':
        t = t + torch.randn_like(t) * 0.05
    elif mode == 'brightness_up':
        t = t * 1.10
    elif mode == 'brightness_down':
        t = t * 0.90
    elif mode == 'rotation':
        t = transforms.functional.rotate(t.squeeze(0), 5).unsqueeze(0)
    return torch.clamp(t, -3, 3)

def tensor_to_pil(tensor):
    t = tensor.squeeze(0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = t * std + mean
    t = torch.clamp(t, 0, 1)
    return transforms.ToPILImage()(t)

def save_temp_image(pil_img, path):
    pil_img.save(path)

def compute_stability(model_name, method, img_name):
    temp_path = os.path.join(IMAGE_DIR, '_temp_.png')
    perturbation_modes = ['gaussian', 'brightness_up', 'brightness_down', 'rotation']

    original_tensor = load_image_tensor(img_name)
    original_maps = generate_xai_maps(model_name, img_name)

    if method not in original_maps:
        return None

    original_map = original_maps[method]
    if original_map.shape != (224, 224):
        return None

    ssim_scores = []
    spearman_scores = []

    for mode in perturbation_modes:
        perturbed_tensor = perturb(original_tensor, mode)
        perturbed_pil = tensor_to_pil(perturbed_tensor)
        save_temp_image(perturbed_pil, temp_path)

        try:
            perturbed_maps = generate_xai_maps(model_name, '_temp_.png')
            perturbed_map = perturbed_maps[method]
            if perturbed_map.shape != (224, 224):
                continue

            s = ssim(original_map, perturbed_map, data_range=1.0)
            r, _ = spearmanr(original_map.flatten(), perturbed_map.flatten())
            if np.isnan(s) or np.isnan(r):
                continue
            ssim_scores.append(s)
            spearman_scores.append(r)
        except Exception as e:
            continue

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not ssim_scores:
        return None

    stability = (np.mean(ssim_scores) + np.mean(spearman_scores)) / 2
    return float(stability)


if __name__ == '__main__':
    import pandas as pd

    DATA_DIR = r'C:\Users\NMAMIT\cti_project\data'
    XAI_DIR  = r'C:\Users\NMAMIT\cti_project\xai_maps'

    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]

    # Test on 2 images only — stability is slow
    test_imgs = matches['Image Index'].values[:2]

    for img_name in test_imgs:
        print(f'\nImage: {img_name}')
        for method in ['gradcam', 'gradcampp']:
            score = compute_stability('densenet121', method, img_name)
            print(f'  {method} stability: {score}')