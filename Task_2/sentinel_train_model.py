import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Dataset Preparation Functions
def reconstruct_full_image_from_tiles_in_memory(tile_dir, tile_size=512):
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.jpg')]
    tile_files.sort(key=lambda f: (int(f.split('_')[-2]), int(f.split('_')[-1].replace('.jpg', ''))))

    max_y = max(int(f.split('_')[-2]) for f in tile_files)
    max_x = max(int(f.split('_')[-1].replace('.jpg', '')) for f in tile_files)

    full_height = max_y + tile_size
    full_width = max_x + tile_size

    full_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)

    for tile_file in tile_files:
        y = int(tile_file.split('_')[-2])
        x = int(tile_file.split('_')[-1].replace('.jpg', ''))
        tile_path = os.path.join(tile_dir, tile_file)
        tile = cv2.imread(tile_path)
        full_image[y:y + tile_size, x:x + tile_size] = tile

    return full_image

def get_tci_tile_folders(root_dir):
    tci_folders = []
    for root, dirs, files in os.walk(root_dir):
        if any('TCI' in file and file.endswith('.jpg') for file in files):
            tci_folders.append(root)
    return tci_folders

# Dataset class for handling TCI image pairs
class BalancedTCIImagePairDataset(Dataset):
    def __init__(self, tci_tile_folders, transform=None):
        self.tci_tile_folders = tci_tile_folders
        self.transform = transform
        self.full_tci_images = [reconstruct_full_image_from_tiles_in_memory(folder) for folder in tci_tile_folders]
        self.pairs = self._create_balanced_pairs()

    def _create_balanced_pairs(self):
        pairs = []
        num_images = len(self.full_tci_images)
        for i in range(num_images):
            img1 = self.full_tci_images[i]
            img2 = img1
            pairs.append((img1, img2, 1))

        for i in range(num_images):
            img1 = self.full_tci_images[i]
            img2 = random.choice(self.full_tci_images)
            while np.array_equal(img2, img1):
                img2 = random.choice(self.full_tci_images)
            pairs.append((img1, img2, 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Prepare the dataset and dataloaders
root_dir = '/path'
tci_tile_folders = get_tci_tile_folders(root_dir)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_tci_folders, test_tci_folders = train_test_split(tci_tile_folders, test_size=0.2, random_state=42)
train_dataset = BalancedTCIImagePairDataset(train_tci_folders, transform=transform)
test_dataset = BalancedTCIImagePairDataset(test_tci_folders, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image1, image2):
        features1 = self.feature_extractor(image1)
        features2 = self.feature_extractor(image2)
        diff = torch.abs(features1 - features2)
        x = torch.relu(self.fc1(diff))
        output = torch.sigmoid(self.fc2(x))
        return output

# Training function
def train_siamese_network(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (img1, img2, labels) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_siamese_network(model, train_loader, criterion, optimizer, device, num_epochs=20)

# Save the trained model
def save_siamese_model(model, optimizer, file_path):
    model_state = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(model_state, file_path)
    print(f"Model saved to {file_path}")

save_siamese_model(model, optimizer, '/path/siamese_model.pth')
