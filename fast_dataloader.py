"""
Fast dataloaders backed by the pre-resized memmap cache (see build_cache.py).
Decoding is already done, so __getitem__ only applies augmentation -> the
training pipeline becomes GPU-bound. num_workers=0 (Windows rule).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

DATA_DIR  = r"C:\Users\NMAMIT\cti_project\data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),                       # uint8 HWC -> PIL
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class CachedDataset(Dataset):
    def __init__(self, split, train=False):
        self.labels = np.load(os.path.join(CACHE_DIR, f"{split}_labels.npy"))
        n = len(self.labels)
        self.arr = np.memmap(os.path.join(CACHE_DIR, f"{split}_imgs.dat"),
                             dtype=np.uint8, mode="r", shape=(n, SIZE, SIZE, 3))
        self.transform = train_transform if train else eval_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = np.array(self.arr[i])                # copy out of memmap
        x = self.transform(img)
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return x, y


def get_fast_dataloaders(batch_size=32):
    train_ds = CachedDataset("train", train=True)
    val_ds   = CachedDataset("val",   train=False)
    test_ds  = CachedDataset("test",  train=False)

    # WeightedRandomSampler -> oversample the minority (Cardiomegaly) class
    labels = train_ds.labels.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader
