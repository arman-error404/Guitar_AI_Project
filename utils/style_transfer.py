"""
Lightweight Style Transfer Module for Real-time Guitar Visualization
Supports both GAN-based and filter-based approaches for real-time performance
Optimized for Jetson Nano with conditional style transfer capability
"""
import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict

# Try to import PyTorch, but fall back to OpenCV-only if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, using OpenCV-only style transfer")


if PYTORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """Lightweight residual block for style transfer"""
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    class ConditionalStyleEmbedding(nn.Module):
        """Embedding layer for style conditioning"""
        def __init__(self, num_styles=6, embedding_dim=64):
            super(ConditionalStyleEmbedding, self).__init__()
            self.embedding = nn.Embedding(num_styles, embedding_dim)
            
        def forward(self, style_id):
            return self.embedding(style_id)

    class LightweightStyleTransfer(nn.Module):
        """
        Lightweight style transfer generator (Pix2Pix-like architecture)
        Supports conditional style transfer via style embeddings
        Optimized for real-time inference on Jetson Nano
        """
        def __init__(self, input_channels=3, output_channels=3, num_styles=6, use_conditional=True):
            super(LightweightStyleTransfer, self).__init__()
            self.use_conditional = use_conditional
            self.num_styles = num_styles
            
            # Style embedding for conditional generation
            if use_conditional:
                self.style_embedding = ConditionalStyleEmbedding(num_styles, embedding_dim=64)
                style_channels = 64
            else:
                style_channels = 0
            
            # Encoder (downsampling)
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, 7, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            
            # Conditional injection point
            if use_conditional:
                # Project style embedding to feature channels (will be broadcasted)
                self.style_projection = nn.Sequential(
                    nn.Linear(style_channels, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 128)
                )
            
            # Residual blocks (lightweight)
            self.res_blocks = nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
            )
            
            # Decoder (upsampling)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, output_channels, 7, padding=3),
                nn.Tanh()
            )
            
        def forward(self, x, style_id=None):
            # Encode input
            encoded = self.encoder(x)
            
            # Inject style conditioning if enabled
            if self.use_conditional and style_id is not None:
                style_emb = self.style_embedding(style_id)  # (B, embedding_dim)
                style_proj = self.style_projection(style_emb)  # (B, 128)
                # Reshape to (B, 128, 1, 1) for broadcasting
                b, c, h, w = encoded.shape
                style_proj = style_proj.unsqueeze(-1).unsqueeze(-1)  # (B, 128, 1, 1)
                # Add as bias to each spatial location
                encoded = encoded + style_proj
            
            # Process through residual blocks
            processed = self.res_blocks(encoded)
            
            # Decode to output
            output = self.decoder(processed)
            return output
else:
    # Dummy class if PyTorch not available
    class LightweightStyleTransfer:
        pass
    class ConditionalStyleEmbedding:
        pass


class StyleTransferProcessor:
    """
    Real-time style transfer processor with multiple style presets
    Supports both GAN-based and filter-based approaches
    """
    def __init__(self, device: Optional[str] = None, use_gan: bool = True, 
                 model_dir: str = "models/style_transfer", use_conditional: bool = True):
        if PYTORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.use_gan = use_gan
            self.use_conditional = use_conditional
        else:
            self.device = 'cpu'
            self.use_gan = False
            self.use_conditional = False
            
        self.model = None
        self.models_dict = {}  # For per-style models
        self.current_style = None
        self.input_size = (256, 256)  # Optimized for Jetson Nano
        self.model_dir = model_dir
        
        # Style to ID mapping (excludes "none")
        style_list = [k for k in AVAILABLE_STYLES.keys() if k != "none"]
        self.style_to_id = {style: idx for idx, style in enumerate(style_list)}
        self.id_to_style = {v: k for k, v in self.style_to_id.items()}
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the style transfer model(s)"""
        if not PYTORCH_AVAILABLE or not self.use_gan:
            print("Using OpenCV-based filter approach for style transfer")
            return
            
        try:
            # Try to load conditional model first
            if self.use_conditional:
                model_path = os.path.join(self.model_dir, "conditional_style_transfer.pth")
                if os.path.exists(model_path):
                    self.model = LightweightStyleTransfer(
                        num_styles=6, use_conditional=True
                    ).to(self.device)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    print(f"Loaded conditional GAN model from {model_path}")
                    return
            else:
                # Try to load individual style models
                for style_name, style_id in self.style_to_id.items():
                    model_path = os.path.join(self.model_dir, f"style_{style_name}.pth")
                    if os.path.exists(model_path):
                        model = LightweightStyleTransfer(
                            num_styles=1, use_conditional=False
                        ).to(self.device)
                        model.load_state_dict(torch.load(model_path, map_location=self.device))
                        model.eval()
                        self.models_dict[style_name] = model
                        print(f"Loaded GAN model for style '{style_name}'")
                
                if self.models_dict:
                    print(f"Loaded {len(self.models_dict)} style-specific GAN models")
                    return
            
            # If no pre-trained models found, initialize with random weights
            if self.use_conditional:
                self.model = LightweightStyleTransfer(
                    num_styles=6, use_conditional=True
                ).to(self.device)
                self.model.eval()
                print(f"Initialized conditional GAN model (untrained) on {self.device}")
                print("Note: Model will produce random outputs until trained. Using filter fallback.")
                self.model = None  # Fall back to filters until trained
            else:
                print("No pre-trained GAN models found. Using filter-based approach.")
                
        except Exception as e:
            print(f"Could not initialize GAN model: {e}, using filter-based approach")
            self.model = None
        
    def apply_style(self, frame: np.ndarray, style_name: str) -> np.ndarray:
        """
        Apply style transfer to a frame using GAN if available, else filter-based
        
        Args:
            frame: Input BGR frame (numpy array)
            style_name: Name of the style to apply
            
        Returns:
            Styled frame (BGR format)
        """
        if style_name == "none" or style_name is None:
            return frame
            
        # Resize for processing
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Try GAN-based approach first
        if self.use_gan and PYTORCH_AVAILABLE:
            styled = self._apply_gan_style(frame_resized, style_name)
            if styled is not None:
                # Resize back to original
                styled = cv2.resize(styled, (w, h))
                return styled
        
        # Fallback to filter-based approach
        styled = self._apply_style_transform(frame_resized, style_name)
        
        # Resize back to original
        styled = cv2.resize(styled, (w, h))
        
        return styled
    
    def _apply_gan_style(self, frame: np.ndarray, style_name: str) -> Optional[np.ndarray]:
        """
        Apply style using GAN model
        
        Args:
            frame: Input BGR frame (256x256)
            style_name: Name of the style to apply
            
        Returns:
            Styled frame or None if GAN not available
        """
        if not PYTORCH_AVAILABLE or style_name not in self.style_to_id:
            return None
        
        try:
            # Convert BGR to RGB and normalize
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb).float()
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            frame_tensor = frame_tensor / 255.0
            frame_tensor = (frame_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            frame_tensor = frame_tensor.to(self.device)
            
            # Get style model
            if self.use_conditional and self.model is not None:
                # Use conditional model
                style_id = torch.tensor([self.style_to_id[style_name]], device=self.device)
                with torch.no_grad():
                    styled_tensor = self.model(frame_tensor, style_id)
            elif style_name in self.models_dict:
                # Use style-specific model
                with torch.no_grad():
                    styled_tensor = self.models_dict[style_name](frame_tensor)
            else:
                return None
            
            # Denormalize and convert back to numpy
            styled_tensor = (styled_tensor + 1) / 2  # Denormalize to [0, 1]
            styled_tensor = styled_tensor.clamp(0, 1)
            styled = styled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            styled = (styled * 255).astype(np.uint8)
            
            # Convert RGB to BGR
            styled = cv2.cvtColor(styled, cv2.COLOR_RGB2BGR)
            return styled
            
        except Exception as e:
            print(f"GAN style transfer error: {e}")
            return None
    
    def _apply_style_transform(self, frame: np.ndarray, style_name: str) -> np.ndarray:
        """
        Apply specific style transformation
        Uses optimized color-space and filter-based approaches for real-time performance
        """
        # Convert BGR to RGB for processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if style_name == "neon":
            return self._neon_style(rgb)
        elif style_name == "sketch":
            return self._sketch_style(rgb)
        elif style_name == "cyberpunk":
            return self._cyberpunk_style(rgb)
        elif style_name == "oil_painting":
            return self._oil_painting_style(rgb)
        elif style_name == "watercolor":
            return self._watercolor_style(rgb)
        elif style_name == "vintage":
            return self._vintage_style(rgb)
        else:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    def _neon_style(self, rgb: np.ndarray) -> np.ndarray:
        """Neon/glowing effect"""
        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
        
        # Enhance value (brightness)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        
        # Convert back
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add glow effect using Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (15, 15), 0)
        neon = cv2.addWeighted(enhanced, 0.7, blurred, 0.3, 0)
        
        # Add edge enhancement
        gray = cv2.cvtColor(neon, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        neon = cv2.addWeighted(neon, 0.85, edges_colored, 0.15, 0)
        
        return cv2.cvtColor(neon, cv2.COLOR_RGB2BGR)
    
    def _sketch_style(self, rgb: np.ndarray) -> np.ndarray:
        """Pencil sketch effect"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Invert
        inverted = 255 - gray
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Color dodge blend
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Convert to 3-channel
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        return cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR)
    
    def _cyberpunk_style(self, rgb: np.ndarray) -> np.ndarray:
        """Cyberpunk aesthetic with teal/magenta color grading"""
        # Convert to LAB color space
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Shift color channels for cyberpunk look
        a = np.clip(a.astype(np.float32) + 10, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) - 20, 0, 255).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        cyberpunk = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add film grain
        noise = np.random.normal(0, 5, cyberpunk.shape).astype(np.float32)
        cyberpunk = np.clip(cyberpunk.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Enhance edges
        gray = cv2.cvtColor(cyberpunk, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cyberpunk = cv2.addWeighted(cyberpunk, 0.9, edges_colored, 0.1, 0)
        
        return cv2.cvtColor(cyberpunk, cv2.COLOR_RGB2BGR)
    
    def _oil_painting_style(self, rgb: np.ndarray) -> np.ndarray:
        """Oil painting effect"""
        # Bilateral filter for smoothing while preserving edges
        oil = cv2.bilateralFilter(rgb, 9, 75, 75)
        
        # Enhance colors
        hsv = cv2.cvtColor(oil, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        oil = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add slight texture
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        oil = cv2.filter2D(oil, -1, kernel * 0.1)
        oil = np.clip(oil, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(oil, cv2.COLOR_RGB2BGR)
    
    def _watercolor_style(self, rgb: np.ndarray) -> np.ndarray:
        """Watercolor painting effect"""
        # Multiple bilateral filters for soft look
        watercolor = cv2.bilateralFilter(rgb, 15, 80, 80)
        watercolor = cv2.bilateralFilter(watercolor, 15, 80, 80)
        
        # Soften edges
        watercolor = cv2.GaussianBlur(watercolor, (5, 5), 0)
        
        # Enhance saturation
        hsv = cv2.cvtColor(watercolor, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return cv2.cvtColor(watercolor, cv2.COLOR_RGB2BGR)
    
    def _vintage_style(self, rgb: np.ndarray) -> np.ndarray:
        """Vintage/retro effect"""
        # Sepia tone
        vintage = rgb.copy().astype(np.float32)
        vintage[:, :, 0] = (rgb[:, :, 0] * 0.272 + rgb[:, :, 1] * 0.534 + rgb[:, :, 2] * 0.131)
        vintage[:, :, 1] = (rgb[:, :, 0] * 0.349 + rgb[:, :, 1] * 0.686 + rgb[:, :, 2] * 0.168)
        vintage[:, :, 2] = (rgb[:, :, 0] * 0.393 + rgb[:, :, 1] * 0.769 + rgb[:, :, 2] * 0.189)
        vintage = np.clip(vintage, 0, 255).astype(np.uint8)
        
        # Add vignette
        h, w = vintage.shape[:2]
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = mask / mask.max()
        mask = 1 - mask * 0.5
        vintage = (vintage * mask[:, :, np.newaxis]).astype(np.uint8)
        
        # Add slight noise
        noise = np.random.normal(0, 3, vintage.shape).astype(np.float32)
        vintage = np.clip(vintage.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(vintage, cv2.COLOR_RGB2BGR)


# Available styles
AVAILABLE_STYLES = {
    "none": "Original",
    "neon": "Neon",
    "sketch": "Sketch",
    "cyberpunk": "Cyberpunk",
    "oil_painting": "Oil Painting",
    "watercolor": "Watercolor",
    "vintage": "Vintage"
}
