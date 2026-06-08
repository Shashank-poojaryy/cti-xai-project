import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def get_dataloaders(data_dir, image_dir, batch_size=32):
    train_dataset = ChestXrayDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        image_dir=image_dir,
        transform=train_transform
    )
    val_dataset = ChestXrayDataset(
        csv_path=os.path.join(data_dir, 'val.csv'),
        image_dir=image_dir,
        transform=val_test_transform
    )
    test_dataset = ChestXrayDataset(
        csv_path=os.path.join(data_dir, 'test.csv'),
        image_dir=image_dir,
        transform=val_test_transform
    )

    # Weighted sampler for class imbalance
    train_labels = train_dataset.data['label'].values
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("DataLoader code is ready.")
    print("Waiting for images to test fully.")
    print("Class weights will be: Cardiomegaly =", round(60361/2776, 1))