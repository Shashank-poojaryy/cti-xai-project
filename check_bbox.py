import os
import pandas as pd

DATA_DIR  = r'C:\Users\NMAMIT\cti_project\data'
IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'

bbox_df = pd.read_csv(os.path.join(DATA_DIR, 'BBox_List_2017.csv'))
bbox_cardio = bbox_df[bbox_df['Finding Label'] == 'Cardiomegaly']

available = set(os.listdir(IMAGE_DIR))

matches = bbox_cardio[bbox_cardio['Image Index'].isin(available)]
print(f"Total Cardiomegaly bbox annotations: {len(bbox_cardio)}")
print(f"Available in our image folder: {len(matches)}")
print("\nMatching images:")
print(matches['Image Index'].values)