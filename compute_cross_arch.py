import os
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

XAI_DIR    = r'C:\Users\NMAMIT\cti_project\xai_maps'
MODEL_NAMES = ['densenet121', 'resnet50', 'efficientnet_b4']

def compute_cross_arch(method, img_id):
    maps = {}
    for model_name in MODEL_NAMES:
        path = os.path.join(XAI_DIR, model_name, f'{img_id}_{method}.npy')
        if os.path.exists(path):
            arr = np.load(path).flatten()
            if len(arr) == 224 * 224:
                maps[model_name] = arr

    if len(maps) < 2:
        return None

    model_list = list(maps.keys())
    pairs = []
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            r, _ = spearmanr(maps[model_list[i]], maps[model_list[j]])
            if not np.isnan(r):
                pairs.append(r)

    return float(np.mean(pairs)) if pairs else None


if __name__ == '__main__':
    import pandas as pd
    import os

    DATA_DIR  = r'C:\Users\NMAMIT\cti_project\data'
    IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'

    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]

    test_imgs = matches['Image Index'].values[:3]

    for img_name in test_imgs:
        img_id = img_name.replace('.png', '')
        print(f'\nImage: {img_name}')
        for method in ['gradcam', 'gradcampp']:
            score = compute_cross_arch(method, img_id)
            print(f'  {method} cross-arch: {score}')