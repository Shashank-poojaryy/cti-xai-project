import os
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_DIR = r'F:\cti_images\images'
DATA_DIR  = r'C:\Users\Acer\cti_project\data'

# Get all available images
available = set(os.listdir(IMAGE_DIR))
print(f"Available images: {len(available)}")

# Load full dataset
data = pd.read_csv(os.path.join(DATA_DIR, 'Data_Entry_2017.csv'))
cardio = data[data['Finding Labels'].str.contains('Cardiomegaly')].copy()
normal = data[data['Finding Labels'] == 'No Finding'].copy()
cardio['label'] = 1
normal['label'] = 0
df = pd.concat([cardio, normal], ignore_index=True)

# Filter to only available images
df = df[df['Image Index'].isin(available)]
print(f"Filtered dataset size: {len(df)}")
print(f"Cardiomegaly: {len(df[df['label']==1])}")
print(f"Normal: {len(df[df['label']==0])}")

# Resplit
train, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val, test   = train_test_split(temp, test_size=0.667, stratify=temp['label'], random_state=42)

train.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
val.to_csv(os.path.join(DATA_DIR, 'val.csv'),   index=False)
test.to_csv(os.path.join(DATA_DIR, 'test.csv'),  index=False)

print(f"\nTrain: {len(train)} | Cardiomegaly: {len(train[train['label']==1])}")
print(f"Val:   {len(val)} | Cardiomegaly: {len(val[val['label']==1])}")
print(f"Test:  {len(test)} | Cardiomegaly: {len(test[test['label']==1])}")
print("\nCSVs updated with available images only.")