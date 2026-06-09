content = open('dataloader.py').read() 
content = content.replace('num_workers=2', 'num_workers=0') 
open('dataloader.py', 'w').write(content) 
print('Fixed.') 
