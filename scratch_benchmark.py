import time, torch
from dataloader import get_dataloaders
DATA_DIR  = r'C:\Users\NMAMIT\cti_project\data'
IMAGE_DIR = r'C:\Users\NMAMIT\cti_project\images'
tr, va, te = get_dataloaders(DATA_DIR, IMAGE_DIR, batch_size=32)
dev = torch.device('cuda')
import torchvision.models as M, torch.nn as nn
m = M.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features,1); m=m.to(dev); m.train()
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
crit = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda')
N=60
t0=time.time(); seen=0
for i,(x,y) in enumerate(tr):
    x=x.to(dev,non_blocking=True); y=y.to(dev).unsqueeze(1)
    opt.zero_grad()
    with torch.amp.autocast('cuda'):
        out=m(x); loss=crit(out,y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    seen+=x.size(0)
    if i+1==N: break
dt=time.time()-t0
print(f"{N} batches, {seen} imgs in {dt:.1f}s -> {seen/dt:.1f} img/s, {dt/N*1000:.0f} ms/batch")
print(f"Est. epoch (44017 imgs): {44017/(seen/dt)/60:.1f} min")
