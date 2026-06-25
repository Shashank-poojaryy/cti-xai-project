import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

DATA_DIR = r'C:\Users\NMAMIT\cti_project\data'
XAI_DIR  = r'C:\Users\NMAMIT\cti_project\xai_maps'
IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'

XAI_METHODS = ['gradcam', 'gradcampp', 'scorecam', 'layercam', 'integrated_gradients']

# Load bbox
bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
available = set(os.listdir(IMAGE_DIR))
matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]
print(f"Images to evaluate: {len(matches)}")

def compute_localization(saliency, bbox, img_size=224):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    scale = img_size / 1024
    x1,y1 = int(x*scale), int(y*scale)
    x2,y2 = int((x+w)*scale), int((y+h)*scale)
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img_size,x2), min(img_size,y2)
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
    return (iou + js_div + pointing) / 3, iou, js_div, pointing

def compute_cross_xai(model_name, img_id):
    maps = {}
    for m in XAI_METHODS:
        path = os.path.join(XAI_DIR, model_name, f'{img_id}_{m}.npy')
        if os.path.exists(path):
            arr = np.load(path).flatten()
            if len(arr) == 224*224:  # only use full size maps
                maps[m] = arr
    methods = list(maps.keys())
    pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            r, _ = spearmanr(maps[methods[i]], maps[methods[j]])
            pairs.append(r)
    return np.mean(pairs) if pairs else 0

results = []
for _, row in matches.iterrows():
    img_name = row['Image Index']
    img_id   = img_name.replace('.png', '')
    bbox = (row['Bbox [x'], row['y'], row['w'], row['h]'])

    xai_score = compute_cross_xai('densenet121', img_id)

    for method in XAI_METHODS:
        path = os.path.join(XAI_DIR, 'densenet121', f'{img_id}_{method}.npy')
        if not os.path.exists(path):
            continue
        saliency = np.load(path)
        if saliency.shape != (224, 224):
            continue  # skip malformed maps
        loc, iou, js, pg = compute_localization(saliency, bbox)
        cti = 0.20*loc + 0.20*xai_score
        results.append({
            'image': img_name,
            'method': method,
            'localization': round(loc, 4),
            'iou': round(iou, 4),
            'js_div': round(js, 4),
            'pointing_game': round(pg, 4),
            'cross_xai': round(xai_score, 4),
            'partial_cti': round(cti, 4)
        })

df = pd.DataFrame(results)
print("\nCTI Results per method (Localization + Cross-XAI):")
print(df.groupby('method')[['localization','cross_xai','partial_cti']].mean().round(4))
df.to_csv(r'C:\Users\NMAMIT\cti_project\results\cti_test_results.csv', index=False)
print("\nSaved to results/cti_test_results.csv")