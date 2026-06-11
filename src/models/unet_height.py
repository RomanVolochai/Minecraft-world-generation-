import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=92, out_channels=1):
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        d1 = self.down1(x)
        x = self.pool1(d1)
        
        d2 = self.down2(x)
        x = self.pool2(d2)
        
        d3 = self.down3(x)
        x = self.pool3(d3)
        
        d4 = self.down4(x)
        x = self.pool4(d4)
        
        x = self.bottleneck(x)
        
        x = self.up4(x)
        x = torch.cat([x, d4], dim=1)
        x = self.conv4(x)
        
        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.conv2(x)
        
        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.conv1(x)
        
        return self.final_conv(x)

class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, surface, biomes, heightmaps, surface_classes, biome_classes):
        self.surface = surface
        self.biomes = biomes
        self.heightmaps = heightmaps
        self.surface_classes = surface_classes
        self.biome_classes = biome_classes

    def __len__(self):
        return len(self.surface)

    def __getitem__(self, idx):
        surf = torch.tensor(self.surface[idx], dtype=torch.long)
        bio = torch.tensor(self.biomes[idx], dtype=torch.long)
        
        surf_oh = torch.nn.functional.one_hot(surf, num_classes=self.surface_classes).float()
        surf_oh = surf_oh.permute(2, 0, 1)
        
        bio_oh = torch.nn.functional.one_hot(bio, num_classes=self.biome_classes).float()
        bio_oh = bio_oh.permute(2, 0, 1)
        
        x = torch.cat([surf_oh, bio_oh], dim=0)
        y = torch.tensor(self.heightmaps[idx], dtype=torch.float32).unsqueeze(0)
        
        return x, y

class UNetHeightModel:
    def __init__(self, num_epochs=10, batch_size=64, lr=0.001):
        self.surface_classes = 27
        self.biome_classes = 65
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.net = UNet(in_channels=self.surface_classes + self.biome_classes, out_channels=1).to(self.device)
        self.fitted = False
        
    def _prepare_data(self, surface, biomes, heightmaps):
        return UNetDataset(surface, biomes, heightmaps, self.surface_classes, self.biome_classes)
        
    def fit(self, surface, biomes, heightmaps):
        print(f"Preparing U-Net dataset...")
        dataset = self._prepare_data(surface, biomes, heightmaps)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        print("Starting U-Net Training Loop...")
        for epoch in range(self.num_epochs):
            losses = []
            loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{self.num_epochs}]")
            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.net(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                loop.set_postfix(L1_Loss=np.mean(losses))
                
        self.fitted = True

    def generate(self, surface, biomes):
        if not self.fitted:
            raise ValueError("Model must be fitted before generation.")
            
        self.net.eval()
        with torch.no_grad():
            dataset = self._prepare_data(surface, biomes, np.zeros_like(surface))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            predictions = []
            for x, _ in dataloader:
                x = x.to(self.device)
                pred = self.net(x)
                predictions.append(pred.squeeze(1).cpu().numpy())
                
        self.net.train()
        return np.concatenate(predictions, axis=0)

    def save(self, filepath):
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.net.state_dict(), filepath)
        
    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.fitted = True
