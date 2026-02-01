"""
GAN-based Audio Enhancement Module for Guitar Chord Recognition
Denoises and enhances spectrograms for improved chord classification
"""
import numpy as np
import cv2
from typing import Optional, Tuple
import os

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, audio enhancement will use filter-based approach")


if PYTORCH_AVAILABLE:
    class AudioResidualBlock(nn.Module):
        """Residual block for audio spectrogram processing"""
        def __init__(self, channels):
            super(AudioResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    class SpectrogramEnhancer(nn.Module):
        """
        GAN-based spectrogram enhancer/denoiser
        Takes noisy spectrogram, outputs cleaned version
        U-Net-like architecture for preserving frequency structure
        """
        def __init__(self, input_channels=1, output_channels=1):
            super(SpectrogramEnhancer, self).__init__()
            
            # Encoder (downsampling)
            self.enc1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.enc3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.enc4 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # Bottleneck
            self.bottleneck = nn.Sequential(
                AudioResidualBlock(256),
                AudioResidualBlock(256),
                AudioResidualBlock(256)
            )
            
            # Decoder (upsampling) with skip connections
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),  # 128*2 for skip
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),  # 64*2 for skip
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, output_channels, 3, padding=1),  # 32*2 for skip
                nn.Tanh()  # Output in [-1, 1] range
            )
            
        def forward(self, x):
            # Encoder with skip connections
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            
            # Bottleneck
            b = self.bottleneck(e4)
            
            # Decoder with skip connections
            d4 = self.dec4(b)
            d3 = self.dec3(torch.cat([d4, e3], dim=1))
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            d1 = self.dec1(torch.cat([d2, e1], dim=1))
            
            return d1

    class AudioDiscriminator(nn.Module):
        """PatchGAN discriminator for audio spectrograms"""
        def __init__(self, input_channels=2):  # 1 noisy + 1 clean
            super(AudioDiscriminator, self).__init__()
            
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
        
        def forward(self, noisy, clean):
            # Concatenate noisy and clean spectrograms
            img_input = torch.cat((noisy, clean), 1)
            return self.model(img_input)
else:
    # Dummy classes if PyTorch not available
    class SpectrogramEnhancer:
        pass
    class AudioDiscriminator:
        pass


class AudioEnhancementProcessor:
    """
    Audio enhancement processor using GAN for spectrogram denoising
    """
    def __init__(self, device: Optional[str] = None, use_gan: bool = True,
                 model_dir: str = "models/audio_gan"):
        if PYTORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.use_gan = use_gan
        else:
            self.device = 'cpu'
            self.use_gan = False
            
        self.model = None
        self.model_dir = model_dir
        self.spec_height = 252  # CQT bins
        self.spec_width = 256   # Time frames
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the audio enhancement model"""
        if not PYTORCH_AVAILABLE or not self.use_gan:
            print("Using filter-based audio enhancement (no GAN)")
            return
            
        try:
            model_path = os.path.join(self.model_dir, "spectrogram_enhancer.pth")
            if os.path.exists(model_path):
                self.model = SpectrogramEnhancer().to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Loaded audio enhancement GAN from {model_path}")
            else:
                print(f"No pre-trained audio GAN found at {model_path}")
                print("Audio enhancement will use filter-based approach")
                self.model = None
        except Exception as e:
            print(f"Could not load audio GAN: {e}, using filter-based approach")
            self.model = None
    
    def enhance_spectrogram(self, spectrogram: np.ndarray, 
                           spec_type: str = "cqt") -> np.ndarray:
        """
        Enhance a spectrogram using GAN if available, else filter-based
        
        Args:
            spectrogram: Input spectrogram (H, W) or (H, W, 1)
            spec_type: Type of spectrogram ("cqt" or "stft")
            
        Returns:
            Enhanced spectrogram with same shape
        """
        # Ensure 2D
        if len(spectrogram.shape) == 3:
            spec = spectrogram.squeeze()
        else:
            spec = spectrogram.copy()
        
        original_shape = spec.shape
        
        # Resize to expected dimensions if needed
        # WARNING: Resizing CQT spectrograms can distort frequency structure
        # Only resize if absolutely necessary (should match 252x256 for CQT)
        if spec.shape != (self.spec_height, self.spec_width):
            if spec_type == "cqt":
                print(f"Warning: CQT spectrogram shape {spec.shape} doesn't match expected {(self.spec_height, self.spec_width)}")
                print("Resizing may distort frequency information. Check preprocessing.")
            spec = cv2.resize(spec, (self.spec_width, self.spec_height), interpolation=cv2.INTER_LINEAR)
        
        # Try GAN enhancement first
        if self.use_gan and self.model is not None:
            enhanced = self._apply_gan_enhancement(spec)
            if enhanced is not None:
                # Resize back if needed
                if enhanced.shape != original_shape:
                    enhanced = cv2.resize(enhanced, (original_shape[1], original_shape[0]))
                return enhanced
        
        # Fallback to filter-based enhancement
        enhanced = self._apply_filter_enhancement(spec)
        
        # Resize back if needed
        if enhanced.shape != original_shape:
            enhanced = cv2.resize(enhanced, (original_shape[1], original_shape[0]))
        
        return enhanced
    
    def _apply_gan_enhancement(self, spectrogram: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply GAN-based enhancement to spectrogram
        
        Args:
            spectrogram: Input spectrogram (H, W) in dB scale (raw values)
            
        Returns:
            Enhanced spectrogram in original dB scale, or None if GAN unavailable
        """
        if not PYTORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Store original scale for proper denormalization
            spec_min, spec_max = spectrogram.min(), spectrogram.max()
            
            # MATCH TRAINING PROCESS EXACTLY:
            # Step 1: Normalize to [0, 1] (same as training line 102-106)
            if spec_max - spec_min > 1e-6:
                spec_norm_01 = (spectrogram - spec_min) / (spec_max - spec_min)
            else:
                spec_norm_01 = spectrogram.copy()
            
            # Step 2: Normalize to [-1, 1] for GAN (same as training line 121)
            spec_norm = spec_norm_01 * 2 - 1
            
            # Convert to tensor
            spec_tensor = torch.from_numpy(spec_norm).float()
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            spec_tensor = spec_tensor.to(self.device)
            
            # Apply GAN
            with torch.no_grad():
                enhanced_tensor = self.model(spec_tensor)
                enhanced_tensor = enhanced_tensor.squeeze(0).squeeze(0)  # (H, W)
                enhanced = enhanced_tensor.cpu().numpy()
            
            # Denormalize: MATCH TRAINING PROCESS IN REVERSE
            # Step 1: From [-1, 1] back to [0, 1]
            enhanced_01 = (enhanced + 1) / 2
            
            # Step 2: From [0, 1] back to original dB scale
            enhanced = enhanced_01 * (spec_max - spec_min) + spec_min
            
            return enhanced
            
        except Exception as e:
            print(f"GAN enhancement error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_filter_enhancement(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply filter-based enhancement (fallback method)
        Uses median filtering and contrast enhancement
        """
        # Ensure float32
        enhanced = spectrogram.astype(np.float32)
        
        # Median filter to reduce noise (only if size allows)
        if enhanced.shape[0] > 3 and enhanced.shape[1] > 3:
            enhanced = cv2.medianBlur(enhanced, 3)
        
        # Normalize to [0, 255] for CLAHE if needed
        spec_min, spec_max = enhanced.min(), enhanced.max()
        if spec_max - spec_min > 1e-6:
            # Normalize to [0, 255] for CLAHE
            enhanced_norm = ((enhanced - spec_min) / (spec_max - spec_min) * 255).astype(np.uint8)
            
            # Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_norm = clahe.apply(enhanced_norm)
            
            # Convert back to original scale
            enhanced = (enhanced_norm.astype(np.float32) / 255.0) * (spec_max - spec_min) + spec_min
        
        # Light Gaussian blur to smooth while preserving structure
        if enhanced.shape[0] > 3 and enhanced.shape[1] > 3:
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def add_noise_to_spectrogram(self, spectrogram: np.ndarray, 
                                 noise_level: float = 0.1) -> np.ndarray:
        """
        Add synthetic noise to spectrogram for training
        
        Args:
            spectrogram: Clean spectrogram
            noise_level: Amount of noise to add (0.0 to 1.0)
            
        Returns:
            Noisy spectrogram
        """
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * spectrogram.std(), spectrogram.shape)
        noisy = spectrogram + noise
        
        # Add random frequency masking (simulate microphone issues)
        if np.random.rand() > 0.5:
            mask_height = int(spectrogram.shape[0] * 0.1)
            mask_start = np.random.randint(0, spectrogram.shape[0] - mask_height)
            noisy[mask_start:mask_start+mask_height, :] *= 0.3
        
        # Add random time masking (simulate dropouts)
        if np.random.rand() > 0.5:
            mask_width = int(spectrogram.shape[1] * 0.05)
            mask_start = np.random.randint(0, spectrogram.shape[1] - mask_width)
            noisy[:, mask_start:mask_start+mask_width] *= 0.3
        
        return noisy
