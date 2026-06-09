import os 
files = ['train.py', 'generate_xai.py', 'compute_cti.py', 'results.py'] 
for f in files: 
    content = open(f).read() 
    content = content.replace(r'C:\Users\Acer\cti_project\data\images', r'F:\cti_images\images') 
    open(f, 'w').write(content) 
    print(f'Updated {f}') 
