import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloader
from configs.config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DEFAULT_TAR_PATH, MOVIENET_PATH

def validate_batch(batch, batch_idx=0):
    """
    Audits a single batch for shapes, ranges, and NaNs.
    """
    l_tensor = batch['L']
    ab_tensor = batch['ab']
    
    print(f"\n🔍 Auditing Batch {batch_idx}...")
    
    # 1. Shape Check
    expected_l_shape = (BATCH_SIZE, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    expected_ab_shape = (BATCH_SIZE, 2, IMAGE_SIZE[0], IMAGE_SIZE[1])
    
    if l_tensor.shape == expected_l_shape:
        print(f"✅ L Shape Passed: {l_tensor.shape}")
    else:
        print(f"❌ L Shape Failed: Expected {expected_l_shape}, Got {l_tensor.shape}")
        
    if ab_tensor.shape == expected_ab_shape:
        print(f"✅ ab Shape Passed: {ab_tensor.shape}")
    else:
        print(f"❌ ab Shape Failed: Expected {expected_ab_shape}, Got {ab_tensor.shape}")

    # 2. Range Check
    l_min, l_max = l_tensor.min().item(), l_tensor.max().item()
    ab_min, ab_max = ab_tensor.min().item(), ab_tensor.max().item()
    
    print(f"📊 L Range: [{l_min:.3f}, {l_max:.3f}]")
    print(f"📊 ab Range: [{ab_min:.3f}, {ab_max:.3f}]")
    
    if -1.1 <= l_min and l_max <= 1.1: # Allow small epsilon
        print("✅ L Normalization Passed ([-1, 1])")
    else:
        print("❌ L Normalization Failed!")
        
    if -1.1 <= ab_min and ab_max <= 1.1:
        print("✅ ab Normalization Passed ([-1, 1])")
    else:
        print("❌ ab Normalization Failed!")

    # 3. NaN Check
    if torch.isnan(l_tensor).any() or torch.isnan(ab_tensor).any():
        print("❌ NaN Detected!")
    else:
        print("✅ No NaNs Detected")

    return l_tensor, ab_tensor

def visualize_validation(l_tensor, ab_tensor, output_path="sprint1_validation_grid.png"):
    """
    Generates visual proof of reconstruction.
    """
    print(f"\n🎨 Generating Visual Proof: {output_path}")
    
    # Denormalize
    l_denorm = (l_tensor + 1.0) * 50.0
    ab_denorm = ab_tensor * 128.0
    
    # Reconstruct
    lab = torch.cat([l_denorm, ab_denorm], dim=1)
    rgb = kornia.color.lab_to_rgb(lab)
    
    # Plot first image in batch
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    
    # Input L (Grayscale)
    l_img = l_denorm[0, 0].cpu().numpy() / 100.0
    axes[0].imshow(l_img, cmap='gray')
    axes[0].set_title("Input Grayscale (L)")
    axes[0].axis('off')
    
    # Output RGB
    rgb_img = rgb[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    axes[1].imshow(rgb_img)
    axes[1].set_title("Reconstructed RGB")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_vizualisation/" + output_path)
    print("✅ Visualization Saved")

def stress_test_streaming(loader, num_batches=50):
    """
    Measures throughput of the streaming loader.
    """
    print(f"\n🚀 Starting Stress Test ({num_batches} batches)...")
    start_time = time.time()
    count = 0
    
    try:
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            # Access data to ensure it's loaded
            _ = batch['L'].shape
            count += 1
            print(f"\rProcessed {count}/{num_batches} batches...", end="")
    except Exception as e:
        print(f"\n❌ Stream Error: {e}")
        
    end_time = time.time()
    duration = end_time - start_time
    total_images = count * BATCH_SIZE
    fps = total_images / duration
    
    print(f"\n⏱️  Time: {duration:.2f}s")
    print(f"⚡ Throughput: {fps:.2f} images/sec")
    
    if fps > 5.0: # Arbitrary threshold for "decent" streaming
        print("✅ Streaming Speed Acceptable")
    else:
        print("⚠️  Streaming Speed Low (Check Connection)")

def main():
    load_dotenv()
    hf_token = os.getenv("HF_AUTH_TOKEN")
    
    # 1. Test Local Source (if available)
    print("\n" + "="*40)
    print("🧪 TEST 1: Local MovieNet Pipeline")
    print("="*40)
    
    # Auto-detect tar
    tar_path = DEFAULT_TAR_PATH
    if os.path.exists(MOVIENET_PATH):
        tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
        if tars:
            tar_path = os.path.join(MOVIENET_PATH, tars[0])
            
    if os.path.exists(tar_path):
        local_loader = get_dataloader('local_movienet', tar_path=tar_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        try:
            batch = next(iter(local_loader))
            l, ab = validate_batch(batch)
            visualize_validation(l, ab, "sprint1_local_proof.png")
        except Exception as e:
            print(f"❌ Local Test Failed: {e}")
    else:
        print("⚠️  Skipping Local Test (No tar found)")

    # 2. Test Remote Source (ImageNet Streaming)
    print("\n" + "="*40)
    print("🧪 TEST 2: Remote ImageNet Pipeline (Streaming)")
    print("="*40)
    
    if hf_token:
        remote_loader = get_dataloader('hf_imagenet', hf_token=hf_token, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        # Audit First Batch
        try:
            batch = next(iter(remote_loader))
            l, ab = validate_batch(batch)
            visualize_validation(l, ab, "sprint1_remote_proof.png")
            
            # Stress Test
            stress_test_streaming(remote_loader, num_batches=20) # 20 batches for quick check
            
        except Exception as e:
            print(f"❌ Remote Test Failed: {e}")
    else:
        print("⚠️  Skipping Remote Test (No HF_AUTH_TOKEN found in .env)")

if __name__ == "__main__":
    main()
