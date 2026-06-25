import pandas as pd 
test_df = pd.read_csv(r'C:\Users\NMAMIT\cti_project\data\test.csv') 
cardio_imgs = test_df[test_df['label']==1]['Image Index'].values[:3] 
from generate_xai import generate_xai_maps, save_maps 
for model_name in ['densenet121']: 
    print(f'Generating XAI maps for {model_name}') 
    for img_name in cardio_imgs: 
        print(f'  Processing {img_name}') 
        maps = generate_xai_maps(model_name, img_name) 
        save_maps(maps, model_name, img_name) 
        print(f'  Saved {len(maps)} maps') 
print('Done.') 
