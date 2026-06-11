import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, num_classes, 4, 2, 1, bias=False),
            nn.Softmax(dim=1) # Output probability distribution over classes
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(num_classes, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

class GANModel:
    def __init__(self, latent_dim=100, num_epochs=5, batch_size=32, lr=0.0002):
        self.num_classes = 27
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.netG = Generator(self.latent_dim, self.num_classes).to(self.device)
        self.netD = Discriminator(self.num_classes).to(self.device)
        self.fitted = False
        
    def _to_one_hot(self, patches):
        
        patches_tensor = torch.tensor(patches, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(patches_tensor, num_classes=self.num_classes).float()
        return one_hot.permute(0, 3, 1, 2)
        
    def fit(self, train_patches):
        print(f"Preparing GAN dataset (converting to One-Hot)...")
        if train_patches.shape[1] != 128 or train_patches.shape[2] != 128:
            raise ValueError("GAN currently only supports 128x128 patches for training.")
            
        dataset = self._to_one_hot(train_patches)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        print("Starting GAN Training Loop...")
        for epoch in range(self.num_epochs):
            g_losses = []
            d_losses = []
            
            loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{self.num_epochs}]")
            for i, data in enumerate(loop):
                real_data = data.to(self.device)
                b_size = real_data.size(0)
                
                self.netD.zero_grad()
                label = torch.full((b_size,), 1.0, dtype=torch.float, device=self.device)
                
                output = self.netD(real_data)
                errD_real = criterion(output, label)
                errD_real.backward()
                
                noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.netG(noise)
                label.fill_(0.0)
                
                output = self.netD(fake_data.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                
                errD = errD_real + errD_fake
                optimizerD.step()
                
                self.netG.zero_grad()
                label.fill_(1.0)  # fake labels are real for generator cost
                output = self.netD(fake_data)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()
                
                g_losses.append(errG.item())
                d_losses.append(errD.item())
                
                loop.set_postfix(D_loss=np.mean(d_losses), G_loss=np.mean(g_losses))
                
        self.fitted = True

    def generate(self, size=(128, 128), return_probs=False, **kwargs):
        if not self.fitted:
            raise ValueError("Model must be fitted before generation.")
            
        H, W = size
        if H != 128 or W != 128:
            print(f"Warning: GAN trained on 128x128. Generating 128x128, ignoring requested {size}.")
            
        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            fake_probs = self.netG(noise)  # (1, 24, 128, 128)
            
            # shape (128, 128, 24)
            probs = fake_probs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        self.netG.train()
        
        if return_probs:
            return probs
            
        generated = np.argmax(probs, axis=-1)
        return generated

    def save(self, filepath):
        
        import os
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
        }, filepath)
        
    def load(self, filepath):
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.fitted = True
