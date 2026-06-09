import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from compute_cti import (compute_localization, compute_stability,
                          compute_cross_xai_agreement,
                          compute_cross_arch_agreement,
                          compute_sanity_check, compute_cti)
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
DATA_DIR    = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR   = r'F:\cti_images\images'
RESULTS_DIR = r'C:\Users\Acer\cti_project\results'
XAI_DIR     = r'C:\Users\Acer\cti_project\xai_maps'

MODEL_NAMES = ['densenet121', 'resnet50', 'efficientnet_b4']
XAI_METHODS = ['gradcam', 'gradcampp', 'scorecam', 'layercam', 'integrated_gradients']

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── LOAD BBOX CSV ────────────────────────────────────────
def load_bbox():
    bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
    bbox_df = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
    return bbox_df


# ─── RUN FULL CTI FOR ALL 15 COMBINATIONS ─────────────────
def run_all_cti():
    bbox_df = load_bbox()
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    # Only use images that have bounding box annotations
    bbox_imgs = set(bbox_df['Image Index'].values)
    test_cardio = test_df[
        (test_df['label'] == 1) &
        (test_df['Image Index'].isin(bbox_imgs))
    ]
    print(f"Test images with bbox annotations: {len(test_cardio)}")

    results = []

    for model_name in MODEL_NAMES:
        for method in XAI_METHODS:
            print(f"\nComputing CTI: {model_name} + {method}")

            loc_scores, stab_scores, sanity_scores = [], [], []

            for _, row in test_cardio.iterrows():
                img_name = row['Image Index']
                img_id   = img_name.replace('.png', '')

                # Get bbox
                bbox_row = bbox_df[bbox_df['Image Index'] == img_name].iloc[0]
                bbox = (bbox_row['Bbox [x'], bbox_row['y'],
                        bbox_row['w'], bbox_row['h]'])

                # Load saliency map
                map_path = os.path.join(XAI_DIR, model_name,
                                        f'{img_id}_{method}.npy')
                if not os.path.exists(map_path):
                    continue
                saliency = np.load(map_path)

                # Localization
                loc, iou, js, pg = compute_localization(saliency, bbox)
                loc_scores.append(loc)

                # Stability
                stab = compute_stability(model_name, method, img_name, None)
                stab_scores.append(stab)

                # Sanity Check
                sanity = compute_sanity_check(model_name, method, img_name)
                sanity_scores.append(sanity)

            # Cross-XAI and Cross-Arch (image-level, then average)
            xai_scores, arch_scores = [], []
            for _, row in test_cardio.iterrows():
                img_id = row['Image Index'].replace('.png', '')
                xai_scores.append(
                    compute_cross_xai_agreement(model_name, img_id))
                arch_scores.append(
                    compute_cross_arch_agreement(method, img_id))

            loc  = np.mean(loc_scores)
            stab = np.mean(stab_scores)
            xai  = np.mean(xai_scores)
            arch = np.mean(arch_scores)
            san  = np.mean(sanity_scores)
            cti  = compute_cti(loc, stab, xai, arch, san)

            results.append({
                'model': model_name,
                'method': method,
                'localization': round(loc, 4),
                'stability': round(stab, 4),
                'cross_xai': round(xai, 4),
                'cross_arch': round(arch, 4),
                'sanity': round(san, 4),
                'CTI': round(cti, 4)
            })
            print(f"  CTI = {cti:.4f}")

    return pd.DataFrame(results)


# ─── PLOT CTI HEATMAP ─────────────────────────────────────
def plot_cti_heatmap(df):
    pivot = df.pivot(index='method', columns='model', values='CTI')
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='CTI Score')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticklabels(pivot.index, fontsize=12)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i,j]:.3f}',
                    ha='center', va='center', fontsize=11, fontweight='bold')
    ax.set_title('CTI Scores — 3 Models × 5 XAI Methods', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cti_heatmap.png'), dpi=150)
    print("Heatmap saved.")


# ─── PATHOLOGY SIZE CORRELATION ───────────────────────────
def pathology_size_correlation(df):
    bbox_df = load_bbox()
    bbox_df['area'] = (bbox_df['w'] * bbox_df['h]']).astype(float)

    correlations = []
    for method in XAI_METHODS:
        method_df = df[df['method'] == method]
        avg_cti = method_df['CTI'].mean()
        area = bbox_df['area'].mean()
        correlations.append({'method': method, 'avg_cti': avg_cti,
                             'avg_bbox_area': area})

    corr_df = pd.DataFrame(correlations)
    r, p = spearmanr(corr_df['avg_bbox_area'], corr_df['avg_cti'])
    print(f"\nPathology Size vs CTI: Spearman r = {r:.4f}, p = {p:.4f}")
    corr_df.to_csv(os.path.join(RESULTS_DIR, 'pathology_correlation.csv'),
                   index=False)
    return r, p


# ─── SENSITIVITY ANALYSIS ─────────────────────────────────
def sensitivity_analysis(df):
    configs = {
        'equal':      [0.20, 0.20, 0.20, 0.20, 0.20],
        'loc_heavy':  [0.30, 0.20, 0.15, 0.20, 0.15],
        'stab_heavy': [0.15, 0.30, 0.20, 0.20, 0.15],
    }
    rows = []
    for _, row in df.iterrows():
        entry = {'model': row['model'], 'method': row['method']}
        for config_name, weights in configs.items():
            cti = (weights[0] * row['localization'] +
                   weights[1] * row['stability']    +
                   weights[2] * row['cross_xai']    +
                   weights[3] * row['cross_arch']   +
                   weights[4] * row['sanity'])
            entry[config_name] = round(cti, 4)
        rows.append(entry)

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_analysis.csv'),
                   index=False)
    print("\nSensitivity Analysis:")
    print(sens_df.to_string(index=False))
    return sens_df


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Results script is ready.")
    print("This script will:")
    print("  1. Compute CTI for all 15 model-XAI combinations")
    print("  2. Plot CTI heatmap (3x5 grid)")
    print("  3. Run pathology size vs CTI correlation")
    print("  4. Run sensitivity analysis with 3 weight configs")
    print("\nRun after training and XAI map generation is complete.")