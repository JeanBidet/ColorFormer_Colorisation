import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage import data, transform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cffn import CFFN


def visualize_cffn_real_image():
    # -----------------------------
    # 1. Load a real image
    # -----------------------------
    if HAS_SKIMAGE:
        img = data.astronaut()  # Image RGB réelle
    else:
        raise ImportError("skimage est requis pour ce test (pip install scikit-image)")

    img = transform.resize(img, (128, 128), anti_aliasing=True)

    # Normalisation [0,1]
    img = img.astype(np.float32)

    # (B, H, W, C)
    x = torch.from_numpy(img).unsqueeze(0)

    B, H, W, C = x.shape
    print(f"Image loaded with shape: {x.shape}")

    # -----------------------------
    # 2. Apply CFFN
    # -----------------------------
    cffn = CFFN(dim=C)
    with torch.no_grad():
        y = cffn(x)

    # -----------------------------
    # 3. Convert to numpy
    # -----------------------------
    x_img = x[0].cpu().numpy()
    y_img = y[0].cpu().numpy()
    diff = np.abs(y_img - x_img)

    # Normalisation pour affichage
    y_img = np.clip(y_img, 0, 1)
    diff = diff / (diff.max() + 1e-8)

    # -----------------------------
    # 4. Visualization
    # -----------------------------
    os.makedirs("outputs/cffn_viz", exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(x_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("After CFFN (Random Weights)")
    plt.imshow(y_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Absolute Difference")
    plt.imshow(diff, cmap="inferno")
    plt.axis("off")

    plt.tight_layout()
    output_path = "outputs/cffn_viz/cffn_real_image_debug.png"
    plt.savefig(output_path)
    plt.show()

    print(f"✅ Image sauvegardée dans {output_path}")


if __name__ == "__main__":
    visualize_cffn_real_image()
