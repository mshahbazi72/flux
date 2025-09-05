#!/usr/bin/env python3
"""
FLUX Alignment Analysis - Dataset Loading and Single Denoising Step

This script loads ImageNet data using PyTorch dataset/dataloader and runs
a single denoising step on FLUX-dev model. This serves as the foundation
for the alignment analysis technique described in the implementation plan.
"""

import random
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import tyro

# Import FLUX components
from src.flux.util import load_flow_model, load_ae, load_t5, load_clip
from src.flux.sampling import prepare


@dataclass
class Config:
    """Configuration for FLUX Alignment Analysis."""
    imagenet_path: str
    """Path to ImageNet dataset (should contain val/ subfolder or class folders)"""
    
    batch_size: int = 4
    """Batch size for processing (default: 4)"""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device for computation (cuda/cpu)"""
    
    seed: int = 42
    """Random seed for reproducibility"""
    
    num_workers: int = 4
    """Number of DataLoader workers"""


class AlignmentAnalyzer:
    """
    Analyzer for computing gradient-based alignment between timesteps in FLUX model.
    
    This class implements the gradient computation and alignment analysis technique
    described in the DiffPruning paper for mixture of experts creation.
    """
    
    def __init__(self, model, ae, t5, clip, device: str = "cuda"):
        """
        Initialize the AlignmentAnalyzer with FLUX model components.
        
        Args:
            model: FLUX transformer model
            ae: Autoencoder for image encoding/decoding
            t5: T5 text encoder
            clip: CLIP text encoder
            device: Device for computation (cuda/cpu)
        """
        self.model = model
        self.ae = ae
        self.t5 = t5
        self.clip = clip
        self.device = torch.device(device)
        
        # Set models to train mode for gradient computation
        self.model.train()
        # Keep encoders in eval mode as we don't need their gradients
        self.ae.eval()
        self.t5.eval()
        self.clip.eval()
        
        # Freeze encoder parameters completely to prevent gradient accumulation
        for param in self.ae.parameters():
            param.requires_grad = False
        for param in self.t5.parameters():
            param.requires_grad = False  
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def add_noise_at_timestep(self, clean_images: torch.Tensor, timestep: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create noisy images at a specific timestep using FLUX flow matching interpolation.
        
        FLUX uses linear interpolation: x_t = (1-t) * clean + t * noise
        where t=0 is clean image and t=1 is pure noise.
        
        Args:
            clean_images: Clean encoded images [B, C, H, W]
            timestep: Timestep value in [0, 1] range (0=clean, 1=noise)
            
        Returns:
            Tuple of (noisy_images, noise) where noise is the sampled noise
        """
        # Generate random noise with same shape as images
        noise = torch.randn_like(clean_images, device=self.device)
        
        # FLUX flow matching uses linear interpolation path
        # At timestep=0, image is clean; at timestep=1, image is pure noise
        # x_t = (1-t) * clean + t * noise
        alpha = 1.0 - timestep  # weight for clean image
        noisy_images = alpha * clean_images + timestep * noise
        
        return noisy_images, noise
    
    def compute_single_timestep_gradients(
        self, 
        images: torch.Tensor, 
        timestep: float,
        text_prompt: list[str] | None = None
    ) -> torch.Tensor:
        """
        Compute gradients of flow matching loss at a single timestep.
        
        FLUX uses flow matching where the model predicts velocity field v(x_t, t)
        that transforms from noise (t=1) to clean image (t=0).
        
        Args:
            images: Batch of clean images [B, 3, H, W] in [-1, 1] range
            timestep: Timestep value in [0, 1] range (1=noise, 0=clean)
            text_prompt: Optional text prompts (uses dummy if None)
            
        Returns:
            Flattened gradient vector for the timestep
        """
        batch_size = images.shape[0]
        
        # Use dummy prompts if not provided
        if text_prompt is None:
            text_prompt = ["a high quality photograph"] * batch_size
        
        # Clear any existing gradients
        self.model.zero_grad()
        
        # Encode images with autoencoder (no gradients needed)
        with torch.no_grad():
            encoded_images = self.ae.encode(images.to(self.device))
        
        # Add noise at the specified timestep
        noisy_images, noise = self.add_noise_at_timestep(encoded_images, timestep)
        
        # Prepare model inputs
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, noisy_images, prompt=text_prompt)
        
        # Create timestep and guidance tensors
        timestep_tensor = torch.full((batch_size,), timestep, dtype=torch.bfloat16, device=self.device)
        guidance_tensor = torch.full((batch_size,), 3.5, device=self.device, dtype=torch.bfloat16)
        
        # Forward pass with gradient computation enabled
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            predicted_velocity = self.model(
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                y=inp["vec"],
                timesteps=timestep_tensor,
                guidance=guidance_tensor,
            )
        
        # Compute flow matching loss (MSE between predicted velocity and velocity target)
        # In flow matching: x_t = (1-t)*clean + t*noise, so velocity = dx/dt = noise - clean
        velocity_target = noise - encoded_images
        
        # Convert velocity target to patch tokens to match model output format
        velocity_target = rearrange(velocity_target, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        
        # Cast velocity_target to match predicted_velocity dtype (bfloat16)
        velocity_target = velocity_target.to(predicted_velocity.dtype)
        
        loss = F.mse_loss(predicted_velocity, velocity_target)
        
        # Compute gradients
        loss.backward()
        
        # Collect and flatten gradients from all model parameters
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.flatten()
        self.model.zero_grad()
        return gradients


def get_transforms(target_size: int = 1024):
    """Create transforms for ImageNet data compatible with FLUX."""
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0)  # Normalize to [-1, 1]
    ])


def load_imagenet_dataset(imagenet_path: str, batch_size: int = 4, num_workers: int = 4):
    """Load ImageNet validation dataset with PyTorch DataLoader."""
    print(f"Loading ImageNet dataset from: {imagenet_path}")
    
    # Expand user path and get validation directory
    val_path = Path(imagenet_path).expanduser() / "val"
    print(f"Looking for validation data at: {val_path}")
    
    # Validate dataset structure
    if not val_path.exists():
        raise ValueError(f"ImageNet validation directory not found: {val_path}")
    
    # Check for class directories (ImageNet classes start with 'n')
    class_dirs = [d for d in val_path.iterdir() if d.is_dir() and d.name.startswith('n')]
    if not class_dirs:
        raise ValueError(f"No ImageNet class directories found in: {val_path}")
    
    print(f"Found {len(class_dirs)} class directories")
    
    try:
        transform = get_transforms()
        dataset = ImageFolder(root=str(val_path), transform=transform)
        
        if len(dataset) == 0:
            raise ValueError(f"No images found in {val_path}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ Loaded ImageNet dataset: {len(dataset)} images, {len(dataset.classes)} classes")
        print(f"   DataLoader: batch_size={batch_size}, num_workers={num_workers}")
        
        return dataloader
        
    except Exception as e:
        raise ValueError(f"Failed to create ImageFolder dataset from {val_path}: {e}")


def load_flux_models(device: str = "cuda"):
    """Load all FLUX-dev model components."""
    print(f"Loading FLUX-dev models on device: {device}")
    torch_device = torch.device(device)
    
    # Load model components
    print("Loading transformer...")
    model = load_flow_model("flux-dev", device=torch_device)
    
    print("Loading autoencoder...")
    ae = load_ae("flux-dev", device=torch_device)
    
    print("Loading T5 text encoder...")
    t5 = load_t5(device=torch_device, max_length=512)
    
    print("Loading CLIP text encoder...")
    clip = load_clip(device=torch_device)
    
    print("All FLUX models loaded successfully")
    return model, ae, t5, clip


def run_single_denoising_step(
    model, ae, t5, clip,
    images: torch.Tensor,
    device: str,
    timestep: float | None = None
) -> Dict[str, Any]:
    """Run a single denoising step on a batch of images."""
    batch_size = images.shape[0]
    torch_device = torch.device(device)
    
    # Generate random timestep if not provided
    if timestep is None:
        timestep = random.uniform(0.1, 0.9)
    
    print(f"Running denoising step for batch_size={batch_size}, timestep={timestep:.3f}")
    
    # Move images to device
    images = images.to(torch_device)
    
    # Encode images with autoencoder
    print("Encoding images with autoencoder...")
    with torch.no_grad():
        # Images are already in [-1, 1] range from transforms
        encoded_images = ae.encode(images)
    
    print(f"Encoded image shape: {encoded_images.shape}")
    
    # Prepare dummy text prompt
    dummy_prompt = ["a high quality photograph"] * batch_size
    
    # Prepare inputs using FLUX sampling utilities
    print("Preparing model inputs...")
    inp = prepare(t5, clip, encoded_images, prompt=dummy_prompt)
    
    # Generate timestep tensor
    timestep_tensor = torch.full((batch_size,), timestep, dtype=torch.bfloat16, device=torch_device)
    
    # Prepare guidance (for flux-dev)
    guidance_tensor = torch.full((batch_size,), 3.5, device=torch_device, dtype=torch.bfloat16)
    
    # Run forward pass through model
    print("Running forward pass through FLUX transformer...")
    start_time = time.time()
    
    with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        pred = model(
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"],
            y=inp["vec"],
            timesteps=timestep_tensor,
            guidance=guidance_tensor,
        )
    
    forward_time = time.time() - start_time
    
    print(f"Forward pass completed in {forward_time:.2f}s")
    print(f"Prediction shape: {pred.shape}")
    
    return {
        "timestep": timestep,
        "batch_size": batch_size,
        "forward_time": forward_time,
        "input_shape": inp["img"].shape,
        "prediction_shape": pred.shape,
        "encoded_image_shape": encoded_images.shape
    }


def main(config: Config):
    
    # Set random seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Load ImageNet dataset
    print("Step 1: Loading ImageNet dataset...")
    dataloader = load_imagenet_dataset(
        config.imagenet_path, 
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    print()
    
    # Load FLUX models
    print("Step 2: Loading FLUX models...")
    model, ae, t5, clip = load_flux_models(device=config.device)
    print()
    
    # Get one random batch
    print("Step 3: Getting random batch...")
    batch_images, batch_labels = next(iter(dataloader))
    print(f"Batch shape: {batch_images.shape}")
    print(f"Batch labels: {batch_labels[:5].tolist()}...")  # Show first 5 labels
    print()
    
    # Create AlignmentAnalyzer and compute gradients
    print("Step 4: Creating AlignmentAnalyzer...")
    analyzer = AlignmentAnalyzer(model, ae, t5, clip, device=config.device)
    print()
    
    # Compute gradients for a single timestep
    print("Step 5: Computing gradients for single timestep...")
    test_timestep_1 = 0.0
    test_timestep_2 = 1


    gradients_1 = analyzer.compute_single_timestep_gradients(
        batch_images, 
        test_timestep_1
    )

    gradients_1_cpu = gradients_1
    del gradients_1

    gradients_2 = analyzer.compute_single_timestep_gradients(
        batch_images, 
        test_timestep_2
    )
    
    print()
    print("✅ Gradient computation completed successfully!")        


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)