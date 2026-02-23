"""
Utility script for loading and displaying TIF images from FVC2004 database.
"""
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_data_path():
    """Get the path to the FVC2004 DB1_B directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data" / "FVC2004_Bundle" / "DB1_B"


def load_tif_image(filename):
    """
    Load a TIF image from the FVC2004 DB1_B directory.
    
    Args:
        filename (str): Name of the TIF file (e.g., '101_1.tif')
        
    Returns:
        torch.Tensor: The loaded image as a torch tensor
    """
    data_path = get_data_path()
    image_path = data_path / filename
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path)
    return torch.from_numpy(np.array(image))


def display_tif_image(filename, cmap='gray', figsize=(8, 8)):
    """
    Load and display a TIF image from the FVC2004 DB1_B directory.
    
    Args:
        filename (str): Name of the TIF file (e.g., '101_1.tif')
        cmap (str): Colormap for display (default: 'gray')
        figsize (tuple): Figure size (default: (8, 8))
    """
    image = load_tif_image(filename)
    
    plt.figure(figsize=figsize)
    plt.imshow(image.numpy(), cmap=cmap)
    plt.title(f"Image: {filename}")
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    return image


def list_available_images():
    """List all available TIF images in the FVC2004 DB1_B directory."""
    data_path = get_data_path()
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        return []
    
    tif_files = sorted([f.name for f in data_path.glob("*.tif")])
    return tif_files


if __name__ == "__main__":
    # Example usage
    print("Available images:")
    images = list_available_images()
    print(f"Found {len(images)} images")
    print(f"First few: {images[:5]}")
    
    # Load and display the first image
    if images:
        print(f"\nLoading {images[10]}...")
        img = load_tif_image(images[10])
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Min value: {img.min()}, Max value: {img.max()}")
        
        # Display the image
        display_tif_image(images[10])
