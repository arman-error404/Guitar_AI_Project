"""
Training script for GAN-based style transfer models
Supports both conditional (single model) and per-style (multiple models) training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

# Add project root directory to path (go up one level from vision/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.style_transfer import LightweightStyleTransfer, StyleTransferProcessor, AVAILABLE_STYLES


class StyleTransferDataset(Dataset):
    """Dataset for style transfer training"""
    def __init__(self, content_dir, style_dir=None, transform=None, style_name=None):
        """
        Args:
            content_dir: Directory with content images
            style_dir: Directory with styled images (if None, uses filter-based generation)
            transform: Image transforms
            style_name: Style name for filter-based generation
        """
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform
        self.style_name = style_name
        self.processor = StyleTransferProcessor(use_gan=False) if style_name else None
        
        # Get list of images
        self.image_files = [f for f in os.listdir(content_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load content image
        img_path = os.path.join(self.content_dir, self.image_files[idx])
        content_img = Image.open(img_path).convert('RGB')
        
        # Get style target
        if self.style_dir:
            # Load pre-styled image
            style_path = os.path.join(self.style_dir, self.image_files[idx])
            if os.path.exists(style_path):
                style_img = Image.open(style_path).convert('RGB')
            else:
                # Fallback to filter-based
                style_img = self._apply_filter_style(content_img)
        else:
            # Generate style using filter-based approach
            style_img = self._apply_filter_style(content_img)
        
        # Apply transforms
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        
        return content_img, style_img
    
    def _apply_filter_style(self, img):
        """Apply filter-based style as target"""
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        styled_bgr = self.processor.apply_style(img_bgr, self.style_name)
        styled_rgb = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(styled_rgb)


class Discriminator(nn.Module):
    """PatchGAN discriminator for adversarial training"""
    def __init__(self, input_channels=6):  # 3 content + 3 style
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, img_A, img_B):
        # Concatenate content and style images
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def train_conditional_model(content_dir, output_dir, num_epochs=100, batch_size=4, 
                           lr=0.0002, device='cuda'):
    """Train a conditional GAN model for all styles"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Create datasets for each style
    datasets = {}
    for style_name in AVAILABLE_STYLES.keys():
        if style_name != "none":
            datasets[style_name] = StyleTransferDataset(
                content_dir, transform=transform, style_name=style_name
            )
    
    # Initialize models
    generator = LightweightStyleTransfer(num_styles=6, use_conditional=True).to(device)
    discriminator = Discriminator().to(device)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    lambda_pixel = 100
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training loop
    print("Starting conditional GAN training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        # Train on each style
        style_list = [k for k in AVAILABLE_STYLES.keys() if k != "none"]
        for style_name, dataset in datasets.items():
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            style_id = torch.tensor([style_list.index(style_name)], device=device)
            
            for i, (content, target) in enumerate(tqdm(dataloader, 
                                                      desc=f"Epoch {epoch+1}/{num_epochs} - {style_name}")):
                content = content.to(device)
                target = target.to(device)
                
                # Adversarial ground truths
                valid = torch.ones((content.size(0), 1, 16, 16), requires_grad=False, device=device)
                fake = torch.zeros((content.size(0), 1, 16, 16), requires_grad=False, device=device)
                
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate fake images
                fake_B = generator(content, style_id.expand(content.size(0)))
                
                # Pixel-wise loss
                loss_pixel = criterion_pixel(fake_B, target)
                
                # Adversarial loss
                pred_fake = discriminator(content, fake_B)
                loss_GAN = criterion_GAN(pred_fake, valid)
                
                # Total generator loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel
                loss_G.backward()
                optimizer_G.step()
                
                # -----------------
                #  Train Discriminator
                # -----------------
                optimizer_D.zero_grad()
                
                # Real loss
                pred_real = discriminator(content, target)
                loss_real = criterion_GAN(pred_real, valid)
                
                # Fake loss
                pred_fake = discriminator(content, fake_B.detach())
                loss_fake = criterion_GAN(pred_fake, fake)
                
                # Total discriminator loss
                loss_D = (loss_real + loss_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
                
                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), 
                      os.path.join(output_dir, f"conditional_style_transfer_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint at epoch {epoch+1}")
            print(f"Generator Loss: {epoch_loss_G/len(datasets):.4f}, "
                  f"Discriminator Loss: {epoch_loss_D/len(datasets):.4f}")
    
    # Save final model
    torch.save(generator.state_dict(), 
              os.path.join(output_dir, "conditional_style_transfer.pth"))
    print("Training complete! Model saved.")


def train_per_style_model(content_dir, style_name, output_dir, num_epochs=50, 
                          batch_size=4, lr=0.0002, device='cuda'):
    """Train a separate model for a specific style"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    dataset = StyleTransferDataset(content_dir, transform=transform, style_name=style_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = LightweightStyleTransfer(num_styles=1, use_conditional=False).to(device)
    discriminator = Discriminator().to(device)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    lambda_pixel = 100
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training loop
    print(f"Training model for style: {style_name}")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        for content, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            content = content.to(device)
            target = target.to(device)
            
            # Adversarial ground truths
            valid = torch.ones((content.size(0), 1, 16, 16), requires_grad=False, device=device)
            fake = torch.zeros((content.size(0), 1, 16, 16), requires_grad=False, device=device)
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_B = generator(content)
            loss_pixel = criterion_pixel(fake_B, target)
            pred_fake = discriminator(content, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(content, target)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(content, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: G_loss={epoch_loss_G/len(dataloader):.4f}, "
                  f"D_loss={epoch_loss_D/len(dataloader):.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, f"style_{style_name}.pth")
    torch.save(generator.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # Get project root for default output path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(project_root, "models", "style_transfer")
    
    parser = argparse.ArgumentParser(description="Train style transfer GAN models")
    parser.add_argument("--content_dir", type=str, required=True,
                       help="Directory containing content images")
    parser.add_argument("--output_dir", type=str, default=default_output,
                       help="Output directory for trained models")
    parser.add_argument("--mode", type=str, choices=["conditional", "per_style"], 
                       default="conditional", help="Training mode")
    parser.add_argument("--style", type=str, default=None,
                       help="Style name for per_style mode")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.mode == "conditional":
        train_conditional_model(
            args.content_dir, args.output_dir, args.epochs, 
            args.batch_size, args.lr, args.device
        )
    else:
        if args.style is None:
            print("Error: --style required for per_style mode")
            exit(1)
        train_per_style_model(
            args.content_dir, args.style, args.output_dir,
            args.epochs, args.batch_size, args.lr, args.device
        )
