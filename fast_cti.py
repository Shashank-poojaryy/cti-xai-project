import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from compute_stability import compute_stability
from compute_sanity import compute_sanity
import warnings
warnings.filterwarnings('ignore')

DATA_DIR  = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR = r'F:\cti_images\images'
XAI_DIR   = r'C:\Users\Acer\cti_project\xai_maps'
RESULTS_DIR = r'C:\Users\Acer\cti_project\results'

XAI_METHODS = ['gradcam', 'gradcampp', 'layercam', 'integrated_gradients']

def compute_localization(saliency, bbox, img_size=224):
    x,y,w,h = float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])
    scale = img_size/1024
    x1,y1 = max(0,int(x*scale)),max(0,int(y*scale))
    x2,y2 = min(img_size,int((x+w)*scale)),min(img_size,int((y+h)*scale))
    gt_mask = np.zeros((img_size,img_size))
    gt_mask[y1:y2,x1:x2] = 1
    threshold = np.percentile(saliency,80)
    pred_mask = (saliency>=threshold).astype(float)
    intersection = (pred_mask*gt_mask).sum()
    union = ((pred_mask+gt_mask)>=1).sum()
    iou = intersection/(union+1e-8)
    sal_flat = saliency.flatten()+1e-8
    gt_flat  = gt_mask.flatten()+1e-8
    sal_flat = sal_flat/sal_flat.sum()
    gt_flat  = gt_flat/gt_flat.sum()
    from scipy.spatial.distance import jensenshannon
    js_div = 1 - jensenshannon(sal_flat,gt_flat)
    peak_y,peak_x = np.unravel_index(np.argmax(saliency),saliency.shape)
    pointing = 1.0 if (y1<=peak_y<=y2 and x1<=peak_x<=x2) else 0.0
    return (iou+js_div+pointing)/3

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
        for j in range(i+1,len(methods)):
            r,_ = spearmanr(maps[methods[i]],maps[methods[j]])
            if not np.isnan(r):
                pairs.append(r)
    return np.mean(pairs) if pairs else 0

if __name__ == '__main__':
    bbox_df = pd.read_csv(os.path.join(DATA_DIR,'BBox_List_2017.csv'))
    bbox_cardio = bbox_df[bbox_df['Finding Label']=='Cardiomegaly']
    available = set(os.listdir(IMAGE_DIR))
    matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]

    # Use only 3 images for speed
    matches = matches.head(3)
    print(f"Evaluating {len(matches)} images")

    results = []
    for method in XAI_METHODS:
        print(f"\nMethod: {method}")
        loc_scores,stab_scores,xai_scores,sanity_scores = [],[],[],[]

        for _,row in matches.iterrows():
            img_name = row['Image Index']
            img_id   = img_name.replace('.png','')
            bbox = (row['Bbox [x'], row['y'], row['w'], row['h]'])

            path = os.path.join(XAI_DIR,'densenet121',f'{img_id}_{method}.npy')
            if not os.path.exists(path):
                continue
            saliency = np.load(path)
            if saliency.shape != (224,224):
                continue

            loc = compute_localization(saliency, bbox)
            loc_scores.append(loc)
            print(f"  {img_name} localization: {loc:.4f}")

            stab = compute_stability('densenet121', method, img_name)
            if stab: stab_scores.append(stab)

            xai = compute_cross_xai('densenet121', img_id)
            xai_scores.append(xai)

            sanity = compute_sanity('densenet121', method, img_name)
            if sanity: sanity_scores.append(sanity)

        loc_m   = np.mean(loc_scores)   if loc_scores   else 0
        stab_m  = np.mean(stab_scores)  if stab_scores  else 0
        xai_m   = np.mean(xai_scores)   if xai_scores   else 0
        sanity_m= np.mean(sanity_scores) if sanity_scores else 0
        cti = 0.25*loc_m + 0.25*stab_m + 0.25*xai_m + 0.25*sanity_m

        print(f"  Localization: {loc_m:.4f} | Stability: {stab_m:.4f} | Cross-XAI: {xai_m:.4f} | Sanity: {sanity_m:.4f} | CTI: {cti:.4f}")
        results.append({'method':method,'localization':round(loc_m,4),
                        'stability':round(stab_m,4),'cross_xai':round(xai_m,4),
                        'sanity':round(sanity_m,4),'CTI':round(cti,4)})

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL CTI RESULTS (4 components, laptop subset)")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR,'fast_cti_results.csv'),index=False)
    print("\nSaved to results/fast_cti_results.csv")