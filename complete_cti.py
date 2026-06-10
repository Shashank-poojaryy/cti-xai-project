import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from compute_stability import compute_stability
from compute_sanity import compute_sanity
import warnings
warnings.filterwarnings('ignore')

DATA_DIR    = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR   = r'F:\cti_images\images'
XAI_DIR     = r'C:\Users\Acer\cti_project\xai_maps'
RESULTS_DIR = r'C:\Users\Acer\cti_project\results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ['densenet121', 'resnet50', 'efficientnet_b4']
XAI_METHODS = ['gradcam', 'gradcampp', 'layercam', 'integrated_gradients']

# ── Metric 1: Localization ─────────────────────────────────
def compute_localization(saliency, bbox, img_size=224):
    x,y,w,h = float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])
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
    gt_flat  = gt_mask.flatten()  + 1e-8
    sal_flat /= sal_flat.sum()
    gt_flat  /= gt_flat.sum()
    js_div = 1 - jensenshannon(sal_flat, gt_flat)
    peak_y, peak_x = np.unravel_index(np.argmax(saliency), saliency.shape)
    pointing = 1.0 if (y1 <= peak_y <= y2 and x1 <= peak_x <= x2) else 0.0
    return (iou + js_div + pointing) / 3, iou, js_div, pointing

# ── Metric 3: Cross-XAI Agreement ─────────────────────────
def compute_cross_xai(model_name, img_id):
    maps = {}
    for m in XAI_METHODS:
        path = os.path.join(XAI_DIR, model_name, f'{img_id}_{m}.npy')
        if os.path.exists(path):
            arr = np.load(path).flatten()
            if len(arr) == 224 * 224:
                maps[m] = arr
    methods = list(maps.keys())
    pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            r, _ = spearmanr(maps[methods[i]], maps[methods[j]])
            if not np.isnan(r):
                pairs.append(r)
    return float(np.mean(pairs)) if pairs else 0.0

# ── Metric 4: Cross-Architecture Agreement ────────────────
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
        for j in range(i+1, len(model_list)):
            r, _ = spearmanr(maps[model_list[i]], maps[model_list[j]])
            if not np.isnan(r):
                pairs.append(r)
    return float(np.mean(pairs)) if pairs else None

# ── CTI Formula ───────────────────────────────────────────
def compute_cti(loc, stab, xai, arch, sanity):
    return (0.20 * loc +
            0.20 * stab +
            0.15 * xai +
            0.20 * arch +
            0.15 * sanity)

# ── Main ──────────────────────────────────────────────────
if __name__ == '__main__':
    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)].head(3)
    print(f"Evaluating {len(matches)} images — all 5 CTI components\n")

    results = []

    for method in XAI_METHODS:
        print(f"{'='*50}")
        print(f"Method: {method.upper()}")
        loc_s, stab_s, xai_s, arch_s, sanity_s = [], [], [], [], []

        for _, row in matches.iterrows():
            img_name = row['Image Index']
            img_id   = img_name.replace('.png', '')
            bbox = (row['Bbox [x'], row['y'], row['w'], row['h]'])

            path = os.path.join(XAI_DIR, 'densenet121', f'{img_id}_{method}.npy')
            if not os.path.exists(path):
                continue
            saliency = np.load(path)
            if saliency.shape != (224, 224):
                continue

            # 1. Localization
            loc, iou, js, pg = compute_localization(saliency, bbox)
            loc_s.append(loc)

            # 2. Stability
            stab = compute_stability('densenet121', method, img_name)
            if stab is not None:
                stab_s.append(stab)

            # 3. Cross-XAI Agreement
            xai = compute_cross_xai('densenet121', img_id)
            xai_s.append(xai)

            # 4. Cross-Architecture Agreement (uses all 3 models)
            arch = compute_cross_arch(method, img_id)
            if arch is not None:
                arch_s.append(arch)

            # 5. Sanity Check
            sanity = compute_sanity('densenet121', method, img_name)
            if sanity is not None:
                sanity_s.append(sanity)

            stab_v   = stab   if stab   is not None else 0.0
            arch_v   = arch   if arch   is not None else 0.0
            sanity_v = sanity if sanity is not None else 0.0
            print(f"  {img_name}: loc={loc:.3f} stab={stab_v:.3f} arch={arch_v:.3f} sanity={sanity_v:.3f}")

        loc_m    = float(np.mean(loc_s))    if loc_s    else 0.0
        stab_m   = float(np.mean(stab_s))   if stab_s   else 0.0
        xai_m    = float(np.mean(xai_s))    if xai_s    else 0.0
        arch_m   = float(np.mean(arch_s))   if arch_s   else 0.0
        sanity_m = float(np.mean(sanity_s)) if sanity_s else 0.0

        cti = compute_cti(loc_m, stab_m, xai_m, arch_m, sanity_m)

        print(f"\n  Localization  : {loc_m:.4f}")
        print(f"  Stability     : {stab_m:.4f}")
        print(f"  Cross-XAI     : {xai_m:.4f}")
        print(f"  Cross-Arch ★  : {arch_m:.4f}")
        print(f"  Sanity        : {sanity_m:.4f}")
        print(f"  ─────────────────────────")
        print(f"  CTI (all 5)   : {cti:.4f}\n")

        results.append({
            'method': method,
            'localization': round(loc_m, 4),
            'stability': round(stab_m, 4),
            'cross_xai': round(xai_m, 4),
            'cross_arch': round(arch_m, 4),
            'sanity': round(sanity_m, 4),
            'CTI': round(cti, 4)
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'complete_cti_results.csv'), index=False)

    print('=' * 50)
    print('FINAL COMPLETE CTI TABLE — All 5 Components')
    print('=' * 50)
    print(df.to_string(index=False))
    print('\nSaved to results/complete_cti_results.csv')

    # Sensitivity analysis — does ranking change with different weights?
    print('\n--- Sensitivity Analysis ---')
    weight_configs = {
        'Equal (0.20 each)':   [0.20, 0.20, 0.15, 0.20, 0.15],
        'Loc-heavy (0.30)':    [0.30, 0.20, 0.10, 0.25, 0.15],
        'Arch-heavy (0.30)':   [0.15, 0.20, 0.10, 0.30, 0.25],
    }
    for config_name, w in weight_configs.items():
        ranked = sorted(results,
                        key=lambda r: w[0]*r['localization'] + w[1]*r['stability'] +
                                      w[2]*r['cross_xai']   + w[3]*r['cross_arch'] +
                                      w[4]*r['sanity'],
                        reverse=True)
        print(f"  {config_name}: {' > '.join([r['method'] for r in ranked])}")