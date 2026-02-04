"""
Diffusion Model Utility for Image Generation
Uses Hugging Face Diffusers DiffusionPipeline.

Default model: OFA-Sys/small-stable-diffusion-v0
Apple devices: runs on MPS with bfloat16 (when available).
"""
import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import random
from typing import Optional
from datetime import datetime


class DiffusionGenerator:
    """Wrapper for a Diffusers text-to-image pipeline."""
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize the diffusion model
        
        Args:
            hf_token: Hugging Face token for authentication (optional, can use HF_TOKEN env var)
            device: Device to run on ("auto", "cuda", "mps", or "cpu")
        """
        self.device = device
        self.pipe = None
        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Random prompt templates for variety
        # These templates enhance your query with style descriptors
        # The {query} placeholder gets replaced with whatever you type
        self.prompt_templates = [
            "{query}",  # Use your query exactly as-is
            "{query}, highly detailed, professional photography",
            "{query}, cinematic lighting, 4k quality",
            "{query}, artistic style, vibrant colors",
            "{query}, studio quality, sharp focus",
            "{query}, dramatic composition, high resolution",
            "{query}, photorealistic, professional grade",
            "{query}, beautiful composition, award winning",
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load the Diffusers pipeline."""
        try:
            # Matches the user-provided snippet (model id)
            model_id = os.getenv("DIFFUSION_MODEL_ID", "OFA-Sys/small-stable-diffusion-v0")
            
            # Prepare authentication
            token = self.hf_token
            if token:
                print("Using HF token from environment variable")
            else:
                print("No HF token provided. Will use cached credentials from 'huggingface-cli login' if available.")
            
            print(f"Loading model: {model_id}...")
            
            # Resolve runtime device
            resolved_device = self._resolve_device(self.device)
            self.device = resolved_device

            # Mirror the user snippet:
            # pipe = DiffusionPipeline.from_pretrained(..., dtype=torch.bfloat16)
            # pipe = pipe.to("mps")
            try:
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    token=token if token else None,
                )
            except (TypeError, ValueError, KeyError):
                # Some versions prefer torch_dtype or don't support bfloat16
                print("bfloat16/dtype not supported, trying torch_dtype=torch.bfloat16...")
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        token=token if token else None,
                    )
                except (TypeError, ValueError):
                    print("bfloat16 not supported, falling back to float32...")
                    self.pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        token=token if token else None,
                    )

            # Move to device explicitly (no device_map="mps")
            if resolved_device == "mps":
                self.pipe = self.pipe.to("mps")
            elif resolved_device == "cuda":
                self.pipe = self.pipe.to("cuda")
            else:
                self.pipe = self.pipe.to("cpu")
            
            print(f"✅ Successfully loaded model: {model_id}")
            print(f"✅ Diffusion model loaded on device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading diffusion model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load diffusion model: {model_id}\n"
                f"Error: {str(e)}\n"
                f"Please check:\n"
                f"1. Your internet connection\n"
                f"2. You're logged in with 'huggingface-cli login'\n"
                f"3. The model exists and is accessible\n"
                f"4. You have accepted any required model licenses on Hugging Face"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve requested device string to an available torch device."""
        requested = (device or "auto").lower()
        if requested == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if requested == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "mps":
            return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        return "cpu"
    
    def generate_image(
        self, 
        query: str, 
        use_random_prompt: bool = True,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = ""
    ) -> Image.Image:
        """
        Generate an image from a text prompt
        
        Args:
            query: Base query/prompt for image generation
            use_random_prompt: If True, randomly selects a prompt template
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            negative_prompt: Negative prompt text (can be empty)
        
        Returns:
            PIL Image object
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Build the prompt
        if use_random_prompt and query:
            # Select a random template and format it
            template = random.choice(self.prompt_templates)
            prompt = template.format(query=query)
        else:
            prompt = query if query else ""
        
        print(f"Generating image with prompt: {prompt}")
        
        try:
            # Generate the image
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            image = result.images[0]
            print("✅ Image generated successfully")
            return image
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            raise
    
    def save_image(self, image: Image.Image, output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save generated image to disk
        
        Args:
            image: PIL Image to save
            output_dir: Directory to save the image
            filename: Optional filename (if None, generates timestamp-based name)
        
        Returns:
            Path to saved image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
        
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"✅ Image saved to: {filepath}")
        return filepath


# Global instance (lazy loaded)
_generator_instance = None


def get_generator(hf_token: Optional[str] = None, device: str = "auto") -> DiffusionGenerator:
    """
    Get or create the global diffusion generator instance
    
    Args:
        hf_token: Hugging Face token (only used on first call)
        device: Device to use (only used on first call)
    
    Returns:
        DiffusionGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DiffusionGenerator(hf_token=hf_token, device=device)
    return _generator_instance
