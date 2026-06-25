import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\NMAMIT\cti_project\data\Data_Entry_2017.csv')

# Filter Cardiomegaly (all) and Normal
cardio = data[data['Finding Labels'].str.contains('Cardiomegaly')].copy()
normal = data[data['Finding Labels'] == 'No Finding'].copy()

# Assign binary labels
cardio['label'] = 1
normal['label'] = 0

# Combine
df = pd.concat([cardio, normal], ignore_index=True)
print("Total dataset size:", len(df))
print("Cardiomegaly:", len(df[df['label']==1]))
print("Normal:", len(df[df['label']==0]))

# Split 70/10/20
train, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.667, stratify=temp['label'], random_state=42)

print("\nTrain size:", len(train), "| Cardiomegaly:", len(train[train['label']==1]))
print("Val size:", len(val), "| Cardiomegaly:", len(val[val['label']==1]))
print("Test size:", len(test), "| Cardiomegaly:", len(test[test['label']==1]))

# Save splits
train.to_csv(r'C:\Users\NMAMIT\cti_project\data\train.csv', index=False)
val.to_csv(r'C:\Users\NMAMIT\cti_project\data\val.csv', index=False)
test.to_csv(r'C:\Users\NMAMIT\cti_project\data\test.csv', index=False)

print("\nSplit CSVs saved to data folder.")