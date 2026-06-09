import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

IMAGE_DIR = r'F:\cti_images\images'
XAI_DIR   = r'C:\Users\Acer\cti_project\xai_maps'
RESULTS_DIR = r'C:\Users\Acer\cti_project\results'

def visualize(img_name, model_name='densenet121'):
    img_id = img_name.replace('.png', '')
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert('RGB')
    img = img.resize((224, 224))

    methods = ['gradcam', 'gradcampp', 'layercam', 'integrated_gradients', 'scorecam']
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, method in enumerate(methods):
        path = os.path.join(XAI_DIR, model_name, f'{img_id}_{method}.npy')
        saliency = np.load(path)
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].imshow(saliency, cmap='jet', alpha=0.5)
        axes[i+1].set_title(method)
        axes[i+1].axis('off')

    plt.suptitle(f'{model_name} — {img_name}', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{img_id}_heatmaps.png')
    plt.savefig(save_path, dpi=150)
    print(f'Saved: {save_path}')
    plt.show()

if __name__ == '__main__':
    visualize('00000432_000.png')