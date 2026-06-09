import re 
content = open('generate_xai.py').read() 
old = 'scorecam = scorecam.cpu().numpy()\n    scorecam = np.maximum(scorecam, 0)\n    return normalize_map(scorecam)' 
new = 'scorecam = scorecam.cpu().numpy()\n    scorecam = np.maximum(scorecam, 0)\n    import torch.nn.functional as F\n    st = torch.tensor(scorecam).unsqueeze(0).unsqueeze(0).float()\n    su = F.interpolate(st, size=(224,224), mode=\"bilinear\", align_corners=False)\n    return normalize_map(su.squeeze().numpy())' 
content = content.replace(old, new) 
open('generate_xai.py', 'w').write(content) 
print('Done. Has upsample:', 'interpolate' in open('generate_xai.py').read()) 
