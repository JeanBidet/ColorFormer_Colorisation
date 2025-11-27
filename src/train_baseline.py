import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast # Deprecated
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import kornia
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import ColorizationUNet
from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE, NUM_WORKERS, DEFAULT_TAR_PATH, MOVIENET_PATH

def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint saved to {path}")

def visualize_training_sample(l_input, ab_pred, ab_target, epoch, output_dir="training_viz"):
    """
    Saves a grid of Input | Prediction | Ground Truth for the first image in batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Take first image
    l = l_input[0].detach().cpu()
    pred = ab_pred[0].detach().cpu()
    target = ab_target[0].detach().cpu()
    
    # Denormalize
    l_denorm = (l + 1.0) * 50.0
    pred_denorm = pred * 128.0
    target_denorm = target * 128.0
    
    # Lab -> RGB
    lab_pred = torch.cat([l_denorm, pred_denorm], dim=0)
    lab_target = torch.cat([l_denorm, target_denorm], dim=0)
    
    rgb_pred = kornia.color.lab_to_rgb(lab_pred).permute(1, 2, 0).numpy()
    rgb_target = kornia.color.lab_to_rgb(lab_target).permute(1, 2, 0).numpy()
    l_img = l_denorm[0].numpy() / 100.0
    
    # Clip
    rgb_pred = np.clip(rgb_pred, 0, 1)
    rgb_target = np.clip(rgb_target, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(l_img, cmap='gray')
    axes[0].set_title("Input (Gray)")
    axes[0].axis('off')
    
    axes[1].imshow(rgb_pred)
    axes[1].set_title(f"Prediction (Epoch {epoch})")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_target)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}.png"))
    plt.close()

def train(args):
    print(f"🚀 Starting Baseline Training on {args.source}...")
    print(f"Subset Limit: {args.limit} samples")
    print(f"Device: {args.device}")
    
    # 1. Data
    use_pin_memory = args.device == 'cuda'
    train_loader = get_dataloader(
        source=args.source,
        split='train',
        limit=args.limit,
        tar_path=args.tar_path,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    
    # 2. Model
    model = ColorizationUNet().to(args.device)
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    
    # Mixed Precision (AMP)
    # Use 'cuda' if available, else 'cpu' (though GradScaler mostly for CUDA)
    use_amp = args.device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_loss = float('inf')
    
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            
            for i, batch in enumerate(pbar):
                l_input = batch['L'].to(args.device)
                ab_target = batch['ab'].to(args.device)
                
                optimizer.zero_grad()
                
                # Forward with AMP
                with torch.amp.autocast('cuda', enabled=use_amp):
                    ab_pred = model(l_input)
                    loss = criterion(ab_pred, ab_target)
                
                # Backward with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                
                # Logging
                if i % 50 == 0:
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = running_loss / (i + 1) if (i + 1) > 0 else 0
            print(f"Epoch {epoch} | Average L1 Loss: {avg_loss:.4f}")
            
            # Save Best Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch, best_loss, "checkpoints/baseline_unet_best.pth")
            
            # Visualization
            visualize_training_sample(l_input, ab_pred, ab_target, epoch)
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        save_checkpoint(model, optimizer, epoch, running_loss, "checkpoints/interrupted.pth")
        
    print("✅ Training Complete.")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="local_movienet")
    parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64) # Fallback to 32 if OOM
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    args.hf_token = os.getenv("HF_AUTH_TOKEN")
    
    # Auto-detect tar
    if args.source == 'local_movienet' and args.tar_path == DEFAULT_TAR_PATH:
        if os.path.exists(MOVIENET_PATH):
            tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
            if tars:
                args.tar_path = os.path.join(MOVIENET_PATH, tars[0])
                
    train(args)
