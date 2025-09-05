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

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import tyro

# Import FLUX components
from src.flux.util import load_flow_model, load_ae, load_t5, load_clip
from src.flux.sampling import prepare
from src.flux.modules.layers import timestep_embedding


@dataclass
class Config:
    """Configuration for FLUX Alignment Analysis."""
    imagenet_path: str
    """Path to ImageNet dataset (should contain val/ subfolder or class folders)"""
    
    batch_size: int = 4
    """Batch size for processing (default: 4)"""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device for computation (cuda/cpu)"""
    
    timestep: float | None = None
    """Fixed timestep for denoising (default: random)"""
    
    seed: int = 42
    """Random seed for reproducibility"""
    
    num_workers: int = 4
    """Number of DataLoader workers"""


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
    
    # Use ImageFolder for ImageNet validation set
    # Assumes structure: imagenet_path/val/class_folders/images
    val_path = Path(imagenet_path) / "val"
    if not val_path.exists():
        # Try direct path if val subfolder doesn't exist
        val_path = Path(imagenet_path)
        if not val_path.exists():
            raise ValueError(f"ImageNet path not found: {imagenet_path}")
    
    transform = get_transforms()
    dataset = ImageFolder(root=str(val_path), transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Loaded ImageNet dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    return dataloader


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
    timestep: float = None
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
    timestep_tensor = torch.full((batch_size,), timestep, dtype=torch.float32, device=torch_device)
    
    # Prepare guidance (for flux-dev)
    guidance_tensor = torch.full((batch_size,), 3.5, device=torch_device, dtype=torch.float32)
    
    # Run forward pass through model
    print("Running forward pass through FLUX transformer...")
    start_time = time.time()
    
    with torch.no_grad():
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


def main():
    config = tyro.cli(Config, description="FLUX Alignment Analysis - Dataset Loading Test")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    print("=" * 60)
    print("FLUX Alignment Analysis - Dataset Loading and Denoising Test")
    print("=" * 60)
    print(f"Configuration: {config}")
    print()
    
    try:
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
        
        # Run single denoising step
        print("Step 4: Running single denoising step...")
        results = run_single_denoising_step(
            model, ae, t5, clip,
            batch_images,
            device=config.device,
            timestep=config.timestep
        )

        
        # Print results summary
        print("Results Summary:")
        print("-" * 30)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())