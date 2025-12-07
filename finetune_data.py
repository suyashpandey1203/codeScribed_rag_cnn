import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

BATCH_SIZE = 32 # Smaller batch size because images are bigger now (224x224)

class FER2013TransferDataset(Dataset):
    def __init__(self, csv_file, split='Training', transform=None):
        self.data = pd.read_csv(csv_file)
        if split == 'Training':
            self.data = self.data[self.data['Usage'] == 'Training']
        elif split == 'Validation':
            self.data = self.data[self.data['Usage'] == 'PublicTest']
        elif split == 'Test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']
            
        self.transform = transform
        self.pixels = self.data['pixels'].tolist()
        self.emotions = self.data['emotion'].tolist()

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        pixel_sequence = self.pixels[idx]
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48).astype('uint8')
        face_img = Image.fromarray(face)
        
        # Convert Grayscale to RGB for ResNet
        face_img = face_img.convert('RGB')

        if self.transform:
            face_img = self.transform(face_img)

        label = torch.tensor(self.emotions[idx], dtype=torch.long)
        return face_img, label

def get_finetune_dataloaders(csv_path='fer2013.csv'):
    # Resize to 224x224 (ResNet Standard)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = FER2013TransferDataset(csv_path, split='Training', transform=train_transforms)
        val_dataset = FER2013TransferDataset(csv_path, split='Validation', transform=val_transforms)
        test_dataset = FER2013TransferDataset(csv_path, split='Test', transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None