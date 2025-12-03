import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
try:
    from skimage import data, transform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.attention import LWMHSA, window_partition

def visualize_shift_and_mask():
    global HAS_SKIMAGE
    print("🎨 Visualizing Attention Shift and Mask...")
    
    # 1. Setup Parameters
    H, W = 256, 256
    window_size = 64
    shift_size = 32
    device = torch.device('cpu')
    
    # 2. Load and Prepare Image
    # 2. Load and Prepare Image
    if HAS_SKIMAGE:
        try:
            img = data.astronaut()
            img = transform.resize(img, (H, W), anti_aliasing=True)
        except Exception as e:
            print(f"⚠️ Failed to load skimage data: {e}")
            HAS_SKIMAGE = False
            
    if not HAS_SKIMAGE:
        print("⚠️ skimage not found or failed. Using synthetic pattern.")
        # Fallback to synthetic pattern
        img = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(0, H, 64):
            for j in range(0, W, 64):
                if (i//64 + j//64) % 2 == 0:
                    img[i:i+64, j:j+64] = [1.0, 1.0, 1.0]
        # Add some color to distinguish channels
        img[:, :, 0] *= 0.8
        img[:, :, 1] *= 0.6
    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    # LWMHSA expects (B, H, W, C)
    x = x.permute(0, 2, 3, 1) # (1, H, W, C)
    
    # 3. Create LWMHSA instance to access helper methods
    lwmhsa = LWMHSA(dim=3, num_heads=1, window_size=window_size, shift_size=shift_size)
    
    # 4. Apply Cyclic Shift
    shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # 5. Calculate Mask
    # The mask shape is (nW, N, N) where N = window_size*window_size
    # We want to visualize the mask for a few windows
    attn_mask = lwmhsa.calculate_mask(H, W, device)
    
    # 6. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original Image with Grid
    ax = axes[0, 0]
    ax.imshow(x[0].numpy())
    ax.set_title("Original Image")
    # Draw grid
    for i in range(0, H+1, window_size):
        ax.axhline(i, color='r', linestyle='--', linewidth=1)
        ax.axvline(i, color='r', linestyle='--', linewidth=1)
    
    # Shifted Image with Grid
    ax = axes[0, 1]
    ax.imshow(shifted_x[0].numpy())
    ax.set_title(f"Cyclic Shift (-{shift_size}, -{shift_size})")
    for i in range(0, H+1, window_size):
        ax.axhline(i, color='r', linestyle='--', linewidth=1)
        ax.axvline(i, color='r', linestyle='--', linewidth=1)
        
    # Highlight the windows that are composed of non-contiguous parts
    # (Top-left, Top-right, Bottom-left, Bottom-right in the shifted view usually correspond to mixed parts)
    
    # Mask Visualization (First Window - Top Left)
    # Usually this one is contiguous if shift is 0, but with shift it might be mixed?
    # Actually, with cyclic shift:
    # Top-left window in shifted view corresponds to the center of original image (contiguous).
    # Bottom-right window in shifted view corresponds to the wrapped around parts (mixed).
    
    # Let's visualize the mask for the last window (bottom-right) which should be heavily masked
    mask_last = attn_mask[-1].numpy() # (N, N)
    
    ax = axes[0, 2]
    im = ax.imshow(mask_last, cmap='viridis')
    ax.set_title("Attention Mask (Bottom-Right Window)")
    plt.colorbar(im, ax=ax)
    
    # Mask for a Top-Left Window (Should be all 0s / unmasked)
    mask_first = attn_mask[0].numpy()
    ax = axes[1, 0]
    im = ax.imshow(mask_first, cmap='viridis')
    ax.set_title("Attention Mask (Top-Left Window)")
    plt.colorbar(im, ax=ax)
    
    # Mask for Top-Right Window
    # Windows are ordered row-major.
    # Grid is H//window_size x W//window_size = 4x4 = 16 windows.
    # Top-Right is index 3.
    mask_tr = attn_mask[3].numpy()
    ax = axes[1, 1]
    im = ax.imshow(mask_tr, cmap='viridis')
    ax.set_title("Attention Mask (Top-Right Window)")
    plt.colorbar(im, ax=ax)
    
    # Mask for Bottom-Left Window
    # Index 12
    mask_bl = attn_mask[12].numpy()
    ax = axes[1, 2]
    im = ax.imshow(mask_bl, cmap='viridis')
    ax.set_title("Attention Mask (Bottom-Left Window)")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    output_path = "outputs/tests_viz/attention_shift_viz.png"
    os.makedirs("outputs/tests_viz", exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_shift_and_mask()
