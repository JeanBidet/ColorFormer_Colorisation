import torch
import matplotlib.pyplot as plt
import kornia
import numpy as np
import os
import argparse
from dotenv import load_dotenv

from src.models.unet import ColorizationUNet
from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE

def visualize_model_predictions(model, dataloader, device='cpu', num_images=4, output_filename="imagenet_viz.png"):
    print("Fetching batch from ImageNet...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("DataLoader is empty.")
        return
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return

    l_input = batch['L'].to(device)
    ab_target = batch['ab'].to(device)
    original = batch['original'] # RGB [0, 1]
    
    print("Running inference...")
    with torch.no_grad():
        ab_pred = model(l_input)
        
    # Prepare for plotting
    # Denormalize L: [-1, 1] -> [0, 100]
    l_denorm = (l_input + 1.0) * 50.0
    
    # Denormalize ab_pred: [-1, 1] -> [-128, 127]
    ab_pred_denorm = ab_pred * 128.0
    
    # Denormalize ab_target: [-1, 1] -> [-128, 127]
    ab_target_denorm = ab_target * 128.0
    
    # Concatenate to get Lab
    lab_pred = torch.cat([l_denorm, ab_pred_denorm], dim=1)
    lab_target = torch.cat([l_denorm, ab_target_denorm], dim=1)
    
    # Convert to RGB
    rgb_pred = kornia.color.lab_to_rgb(lab_pred)
    rgb_target = kornia.color.lab_to_rgb(lab_target) # Should match 'original' closely
    
    # Clip to [0, 1]
    rgb_pred = torch.clamp(rgb_pred, 0, 1)
    rgb_target = torch.clamp(rgb_target, 0, 1)
    
    # Move to CPU for plotting
    l_denorm = l_denorm.cpu()
    rgb_pred = rgb_pred.cpu()
    rgb_target = rgb_target.cpu()
    
    # Plot
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = [axes]
        
    for i in range(num_images):
        if i >= len(l_input):
            break
            
        # Input L (Grayscale)
        l_img = l_denorm[i, 0].numpy() / 100.0
        
        # Prediction
        pred_img = rgb_pred[i].permute(1, 2, 0).numpy()
        
        # Ground Truth (using the one reconstructed from Lab to be fair comparison of colorization, 
        # or use 'original' for absolute truth. Let's use 'original' if available, else rgb_target)
        gt_img = original[i].permute(1, 2, 0).numpy()
        
        ax_row = axes[i] if num_images > 1 else axes
        
        ax_row[0].imshow(l_img, cmap='gray')
        ax_row[0].set_title("Input (Grayscale)")
        ax_row[0].axis('off')
        
        ax_row[1].imshow(pred_img)
        ax_row[1].set_title("Prediction (Untrained U-Net)")
        ax_row[1].axis('off')
        
        ax_row[2].imshow(gt_img)
        ax_row[2].set_title("Ground Truth")
        ax_row[2].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Visualization saved to {output_filename}")

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    hf_token = os.getenv("HF_AUTH_TOKEN")
    
    # 1. Load Data (ImageNet Validation)
    # Note: We use 'val' split. If streaming fails for val, we might need 'train'.
    # But ImageNet usually has 'validation'.
    print("Loading ImageNet DataLoader...")
    loader = get_dataloader(
        source='hf_imagenet',
        split='val',
        hf_token=hf_token,
        batch_size=args.num_images, # Just get enough for viz
        num_workers=0 # Avoid multiprocessing issues for simple script
    )
    
    # 2. Load Model
    print("Initializing Model...")
    model = ColorizationUNet().to(args.device)
    model.eval()
    
    # 3. Visualize
    visualize_model_predictions(
        model=model,
        dataloader=loader,
        device=args.device,
        num_images=args.num_images,
        output_filename="imagenet_viz.png"
    )
