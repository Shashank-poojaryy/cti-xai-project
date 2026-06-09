from dataloader import get_dataloaders

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(
        r'C:\Users\Acer\cti_project\data',
        r'F:\cti_images\images',
        batch_size=32
    )

    images, labels = next(iter(train_loader))
    print('Batch shape:', images.shape)
    print('Labels shape:', labels.shape)
    print('Cardiomegaly in batch:', labels.sum().item())
    print('DataLoader working correctly.')