import torch
import matplotlib.pyplot as plt
import kornia
import numpy as np
import tarfile
import io
from PIL import Image
import os


def visualize_batch(dataloader, output_filename="visualization_output.png"):
    """
    Fetches a batch and visualizes the Input (L) vs Output (Reconstructed RGB).
    """
    try:
        print("Fetching batch...")
        batch = next(iter(dataloader))
    except StopIteration:
        print("DataLoader is empty.")
        return
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return

    l_norm = batch['L'] # Luminance
    ab_norm = batch['ab'] # Chrominance
    original = batch['original'] # Original RGB
    
    batch_size = l_norm.size(0)
    print(f"Batch size: {batch_size}")
    print(f"L shape: {l_norm.shape}, Range: [{l_norm.min():.2f}, {l_norm.max():.2f}]")
    print(f"ab shape: {ab_norm.shape}, Range: [{ab_norm.min():.2f}, {ab_norm.max():.2f}]")

    # De-normalization
    # L: [-1, 1] -> [0, 100]
    l_denorm = (l_norm + 1.0) * 50.0
    
    # ab: [-1, 1] -> [-128, 127]
    ab_denorm = ab_norm * 128.0
    
    # Concatenate L and ab to get Lab
    # L is (B, 1, H, W), ab is (B, 2, H, W)
    lab_tensor = torch.cat([l_denorm, ab_denorm], dim=1)
    
    # Convert Lab -> RGB using Kornia
    rgb_reconstructed = kornia.color.lab_to_rgb(lab_tensor)
    
    # Plotting
    num_viz = min(4, batch_size)
    fig, axes = plt.subplots(num_viz, 3, figsize=(15, 5 * num_viz))
    
    if num_viz == 1:
        axes = [axes] # Handle single image case
    elif batch_size > 1 and isinstance(axes, np.ndarray) and len(axes.shape) == 1:
         # Handle case where subplots returns 1D array if num_viz > 1 but rows=1? 
         # No, (num_viz, 3) ensures 2D unless num_viz=1.
         pass

    # Ensure axes is iterable as list of rows
    if num_viz == 1:
        axes = [axes]
    
    for i in range(num_viz):
        # Original RGB
        orig_img = original[i].permute(1, 2, 0).numpy()
        
        # Grayscale Input (L channel)
        # L is [0, 100], normalize to [0, 1] for display
        l_img = l_denorm[i, 0].numpy() / 100.0
        
        # Reconstructed RGB
        recon_img = rgb_reconstructed[i].permute(1, 2, 0).numpy()
        # Clip to [0, 1] just in case
        recon_img = np.clip(recon_img, 0, 1)

        axes[i][0].imshow(orig_img)
        axes[i][0].set_title("Original RGB")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(l_img, cmap='gray')
        axes[i][1].set_title("Input L (Grayscale)")
        axes[i][1].axis('off')
        
        axes[i][2].imshow(recon_img)
        axes[i][2].set_title("Reconstructed RGB (from Lab)")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Visualization saved to {output_filename}")
