"""
Training script for GAN-based audio spectrogram enhancement
Trains a denoising GAN to improve chord recognition accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import cv2
import os
import argparse
from tqdm import tqdm
import sys
import soundfile as sf
from glob import glob

# Add project root directory to path (go up one level from audio/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.audio_gan import SpectrogramEnhancer, AudioDiscriminator, AudioEnhancementProcessor


class AudioSpectrogramDataset(Dataset):
    """Dataset for audio spectrogram enhancement training"""
    def __init__(self, audio_dir, transform=None, add_noise=True, noise_level=0.15,
                 sample_rate=22050, duration=3.0, use_cqt=True):
        """
        Args:
            audio_dir: Directory with audio files
            transform: Optional transforms
            add_noise: Whether to add synthetic noise
            noise_level: Level of noise to add
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
            use_cqt: Use CQT (True) or STFT (False)
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_cqt = use_cqt
        self.enhancer = AudioEnhancementProcessor(use_gan=False)
        
        # Get list of audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            self.audio_files.extend(glob(os.path.join(audio_dir, ext)))
            self.audio_files.extend(glob(os.path.join(audio_dir, '**', ext), recursive=True))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                    duration=self.duration, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data
            audio = np.zeros(int(self.sample_rate * self.duration))
        
        # Compute clean spectrogram
        if self.use_cqt:
            # Use CQT (better for music)
            spec = np.abs(librosa.cqt(
                y=audio,
                sr=self.sample_rate,
                bins_per_octave=36,
                n_bins=252
            ))
            spec = librosa.amplitude_to_db(spec, ref=np.max)
        else:
            # Use STFT
            spec = librosa.stft(audio, n_fft=1024, hop_length=256)
            spec = np.abs(spec)
            spec = librosa.amplitude_to_db(spec)
        
        # Pad or truncate to fixed size
        target_height, target_width = 252, 256
        if spec.shape[0] < target_height:
            spec = np.pad(spec, ((0, target_height - spec.shape[0]), (0, 0)))
        else:
            spec = spec[:target_height, :]
        
        if spec.shape[1] < target_width:
            spec = np.pad(spec, ((0, 0), (0, target_width - spec.shape[1])))
        else:
            spec = spec[:, :target_width]
        
        # Normalize to [0, 1] for training
        spec_min, spec_max = spec.min(), spec.max()
        if spec_max - spec_min > 1e-6:
            spec_norm = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = spec.copy()
        
        # Create noisy version
        if self.add_noise:
            noisy_spec = self.enhancer.add_noise_to_spectrogram(
                spec_norm, noise_level=self.noise_level
            )
        else:
            noisy_spec = spec_norm.copy()
        
        # Convert to tensor
        clean_tensor = torch.from_numpy(spec_norm).float().unsqueeze(0)  # (1, H, W)
        noisy_tensor = torch.from_numpy(noisy_spec).float().unsqueeze(0)  # (1, H, W)
        
        # Normalize to [-1, 1] for GAN
        clean_tensor = clean_tensor * 2 - 1
        noisy_tensor = noisy_tensor * 2 - 1
        
        return noisy_tensor, clean_tensor


def train_audio_gan(audio_dir, output_dir, num_epochs=100, batch_size=8,
                   lr=0.0002, device='cuda', use_cqt=True):
    """Train audio spectrogram enhancement GAN"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = AudioSpectrogramDataset(
        audio_dir, add_noise=True, noise_level=0.15, use_cqt=use_cqt
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize models
    generator = SpectrogramEnhancer().to(device)
    discriminator = AudioDiscriminator().to(device)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    lambda_pixel = 100  # Weight for pixel loss
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    print("Starting audio GAN training...")
    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"Dataset size: {len(dataset)}")
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        for batch_idx, (noisy, clean) in enumerate(tqdm(dataloader, 
                                                       desc=f"Epoch {epoch+1}/{num_epochs}")):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Adversarial ground truths
            batch_size = noisy.size(0)
            patch_size = 16  # PatchGAN output size
            valid = torch.ones((batch_size, 1, patch_size, patch_size), 
                             requires_grad=False, device=device)
            fake = torch.zeros((batch_size, 1, patch_size, patch_size), 
                              requires_grad=False, device=device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate enhanced spectrogram
            enhanced = generator(noisy)
            
            # Pixel-wise loss (L1)
            loss_pixel = criterion_pixel(enhanced, clean)
            
            # Adversarial loss
            pred_fake = discriminator(noisy, enhanced)
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
            pred_real = discriminator(noisy, clean)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(noisy, enhanced.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Print progress
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Generator Loss: {avg_loss_G:.4f}")
        print(f"  Discriminator Loss: {avg_loss_D:.4f}")
        print(f"  Learning Rate: {scheduler_G.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, 
                                         f"spectrogram_enhancer_epoch_{epoch+1}.pth")
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "spectrogram_enhancer.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    # Get project root for default output path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(project_root, "models", "audio_gan")
    
    parser = argparse.ArgumentParser(description="Train audio spectrogram enhancement GAN")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Directory containing audio files for training")
    parser.add_argument("--output_dir", type=str, default=default_output,
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use_cqt", action="store_true", default=True,
                       help="Use CQT spectrogram (default: True)")
    parser.add_argument("--use_stft", action="store_true", default=False,
                       help="Use STFT spectrogram instead of CQT")
    
    args = parser.parse_args()
    
    use_cqt = args.use_cqt and not args.use_stft
    
    train_audio_gan(
        args.audio_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        use_cqt
    )
