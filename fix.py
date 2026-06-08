content = open('generate_xai.py').read() 
content = content.replace('LayerGradCamPlusPlus, ', '') 
open('generate_xai.py', 'w').write(content) 
