import pandas as pd

data = pd.read_csv(r'C:\Users\NMAMIT\cti_project\data\Data_Entry_2017.csv')
bbox = pd.read_csv(r'C:\Users\NMAMIT\cti_project\data\BBox_List_2017.csv')

cardio = data[data['Finding Labels'] == 'Cardiomegaly']
normal = data[data['Finding Labels'] == 'No Finding']

print("Cardiomegaly cases:", len(cardio))
print("Normal cases:", len(normal))
print("Class imbalance ratio:", round(len(normal)/len(cardio), 1))
print("\nBBox diseases:", bbox['Finding Label'].unique())
print("BBox Cardiomegaly rows:", len(bbox[bbox['Finding Label'] == 'Cardiomegaly']))