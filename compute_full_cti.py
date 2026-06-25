import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity as ssim
from compute_stability import compute_stability
from compute_sanity import compute_sanity
import warnings
warnings.filterwarnings('ignore')

DATA_DIR  = r'C:\Users\NMAMIT\cti_project\data'
IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'
XAI_DIR   = r'C:\Users\NMAMIT\cti_project\xai_maps'
RESULTS_DIR = r'C:\Users\NMAMIT\cti_project\results'

MODEL_NAMES = ['densenet121']
XAI_METHODS = ['gradcam', 'gradcampp', 'scorecam', 'layercam', 'integrated_gradients']

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Localization ───────────────────────────────────────────
def compute_localization(saliency, bbox, img_size=224):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    scale = img_size / 1024
    x1,y1 = max(0,int(x*scale)), max(0,int(y*scale))
    x2,y2 = min(img_size,int((x+w)*scale)), min(img_size,int((y+h)*scale))
    gt_mask = np.zeros((img_size, img_size))
    gt_mask[y1:y2, x1:x2] = 1
    threshold = np.percentile(saliency, 80)
    pred_mask = (saliency >= threshold).astype(float)
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) >= 1).sum()
    iou = intersection / (union + 1e-8)
    sal_flat = saliency.flatten() + 1e-8
    gt_flat  = gt_mask.flatten() + 1e-8
    sal_flat = sal_flat / sal_flat.sum()
    gt_flat  = gt_flat / gt_flat.sum()
    js_div = 1 - jensenshannon(sal_flat, gt_flat)
    peak_y, peak_x = np.unravel_index(np.argmax(saliency), saliency.shape)
    pointing = 1.0 if (y1 <= peak_y <= y2 and x1 <= peak_x <= x2) else 0.0
    return (iou + js_div + pointing) / 3

# ── Cross-XAI Agreement ────────────────────────────────────
def compute_cross_xai(model_name, img_id):
    maps = {}
    for m in XAI_METHODS:
        path = os.path.join(XAI_DIR, model_name, f'{img_id}_{m}.npy')
        if os.path.exists(path):
            arr = np.load(path).flatten()
            if len(arr) == 224*224:
                maps[m] = arr
    methods = list(maps.keys())
    pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            r, _ = spearmanr(maps[methods[i]], maps[methods[j]])
            if not np.isnan(r):
                pairs.append(r)
    return np.mean(pairs) if pairs else 0

# ── Main CTI computation ───────────────────────────────────
def compute_full_cti():
    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]
    print(f"Evaluating {len(matches)} images with bbox annotations")

    results = []

    for model_name in MODEL_NAMES:
        for method in XAI_METHODS:
            print(f"\n{model_name} + {method}")
            loc_scores, stab_scores, xai_scores, sanity_scores = [], [], [], []

            for _, row in matches.iterrows():
                img_name = row['Image Index']
                img_id   = img_name.replace('.png', '')
                bbox = (row['Bbox [x'], row['y'], row['w'], row['h]'])

                # Load saliency map
                path = os.path.join(XAI_DIR, model_name, f'{img_id}_{method}.npy')
                if not os.path.exists(path):
                    continue
                saliency = np.load(path)
                if saliency.shape != (224, 224):
                    continue

                # 1. Localization
                loc = compute_localization(saliency, bbox)
                loc_scores.append(loc)

                # 2. Stability
                stab = compute_stability(model_name, method, img_name)
                if stab is not None:
                    stab_scores.append(stab)

                # 3. Cross-XAI
                xai = compute_cross_xai(model_name, img_id)
                xai_scores.append(xai)

                # 4. Sanity Check
                sanity = compute_sanity(model_name, method, img_name)
                if sanity is not None:
                    sanity_scores.append(sanity)

            loc_mean   = np.mean(loc_scores)   if loc_scores   else 0
            stab_mean  = np.mean(stab_scores)  if stab_scores  else 0
            xai_mean   = np.mean(xai_scores)   if xai_scores   else 0
            sanity_mean= np.mean(sanity_scores) if sanity_scores else 0

            # Note: Cross-Architecture needs 3 models — set to 0 for now
            arch_mean = 0

            cti = (0.20 * loc_mean  +
                   0.20 * stab_mean +
                   0.20 * xai_mean  +
                   0.20 * arch_mean +
                   0.20 * sanity_mean)

            print(f"  Localization:  {loc_mean:.4f}")
            print(f"  Stability:     {stab_mean:.4f}")
            print(f"  Cross-XAI:     {xai_mean:.4f}")
            print(f"  Sanity:        {sanity_mean:.4f}")
            print(f"  CTI (4/5):     {cti:.4f}")

            results.append({
                'model': model_name,
                'method': method,
                'localization': round(loc_mean, 4),
                'stability': round(stab_mean, 4),
                'cross_xai': round(xai_mean, 4),
                'cross_arch': round(arch_mean, 4),
                'sanity': round(sanity_mean, 4),
                'CTI': round(cti, 4)
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'full_cti_results.csv'), index=False)
    print("\n" + "="*50)
    print("FINAL CTI RESULTS")
    print("="*50)
    print(df[['method','localization','stability','cross_xai','sanity','CTI']].to_string(index=False))
    print("\nSaved to results/full_cti_results.csv")
    return df

if __name__ == '__main__':
    compute_full_cti()