import pandas as pd

data = pd.read_csv(r'C:\Users\NMAMIT\cti_project\data\Data_Entry_2017.csv')

# Find ALL rows that contain Cardiomegaly (including combined labels)
cardio_all = data[data['Finding Labels'].str.contains('Cardiomegaly')]
normal = data[data['Finding Labels'] == 'No Finding']

print("Pure Cardiomegaly:", len(data[data['Finding Labels'] == 'Cardiomegaly']))
print("All Cardiomegaly (including combined):", len(cardio_all))
print("Normal cases:", len(normal))
print("\nCombined label examples:")
print(cardio_all['Finding Labels'].value_counts())