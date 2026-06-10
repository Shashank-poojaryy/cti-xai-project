import os
import pandas as pd
from generate_xai import generate_xai_maps, save_maps

DATA_DIR  = r'C:\Users\Acer\cti_project\data'
IMAGE_DIR = r'F:\cti_images\images'

bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']
available = set(os.listdir(IMAGE_DIR))
matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]

print(f"Generating XAI maps for {len(matches)} bbox images...")

for _, row in matches.iterrows():
    img_name = row['Image Index']
    print(f"  Processing {img_name}")
    for model_name in ['densenet121', 'resnet50', 'efficientnet_b4']:
        maps = generate_xai_maps(model_name, img_name)
        save_maps(maps, model_name, img_name)
        print(f"    Saved {len(maps)} maps")

print("Done.")