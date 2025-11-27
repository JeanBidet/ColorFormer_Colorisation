import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import matplotlib.pyplot as plt
import kornia
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from src.models.unet import ColorizationUNet
from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE, NUM_WORKERS, DEFAULT_TAR_PATH, MOVIENET_PATH

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/training_progress", exist_ok=True)

def save_snapshot(model, val_loader, device, epoch, output_dir="outputs/training_progress"):
    """
    Saves a visualization snapshot of the first batch in validation.
    """
    model.eval()
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return
        
    l_input = batch['L'].to(device)
    ab_target = batch['ab'].to(device)
    original = batch['original'] # RGB [0, 1]
    
    with torch.no_grad():
        ab_pred = model(l_input)
        
    # Prepare for plotting (Take first image)
    i = 0
    
    # Denormalize
    l_denorm = (l_input[i] + 1.0) * 50.0
    ab_pred_denorm = ab_pred[i] * 128.0
    ab_target_denorm = ab_target[i] * 128.0
    
    # Lab -> RGB
    lab_pred = torch.cat([l_denorm, ab_pred_denorm], dim=0).unsqueeze(0) # (1, 3, H, W)
    lab_target = torch.cat([l_denorm, ab_target_denorm], dim=0).unsqueeze(0)
    
    rgb_pred = kornia.color.lab_to_rgb(lab_pred).squeeze(0).cpu()
    rgb_target = kornia.color.lab_to_rgb(lab_target).squeeze(0).cpu()
    
    # Clip
    rgb_pred = torch.clamp(rgb_pred, 0, 1)
    rgb_target = torch.clamp(rgb_target, 0, 1)
    
    # Input L
    l_img = l_denorm[0].cpu().numpy() / 100.0
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(l_img, cmap='gray')
    axes[0].set_title("Input (Grayscale)")
    axes[0].axis('off')
    
    axes[1].imshow(rgb_pred.permute(1, 2, 0).numpy())
    axes[1].set_title(f"Prediction (Epoch {epoch})")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_target.permute(1, 2, 0).numpy())
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_snapshot.png"))
    plt.close()
    model.train()

def train(args):
    print(f"🚀 Starting Training on {args.device}...")
    
    # 1. Data
    print("Loading Data...")
    # Auto-detect tar logic
    current_tar = args.tar_path
    if args.source == 'local_movienet':
        if current_tar == DEFAULT_TAR_PATH:
            if os.path.exists(MOVIENET_PATH):
                tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
                if tars:
                    current_tar = os.path.join(MOVIENET_PATH, tars[0])
                    print(f"Auto-detected tar: {current_tar}")

    train_loader = get_dataloader(
        source=args.source,
        split='train',
        tar_path=current_tar,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        source=args.source,
        split='val',
        tar_path=current_tar,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 2. Model
    print("Initializing Model...")
    model = ColorizationUNet(n_classes=2).to(args.device)
    
    # 3. Loss & Optimizer
    # Using L1 Loss (MAE) as it is robust to outliers and produces sharper images than MSE for colorization
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # 4. Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        
        # Train Step
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for i, batch in enumerate(pbar):
            if args.limit and i >= args.limit:
                break
                
            l_input = batch['L'].to(args.device)
            ab_target = batch['ab'].to(args.device)
            
            optimizer.zero_grad()
            
            ab_pred = model(l_input)
            loss = criterion(ab_pred, ab_target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': running_loss / count})
            
        avg_train_loss = running_loss / count if count > 0 else 0
        history['train_loss'].append(avg_train_loss)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        # We limit validation to avoiding long waits on streaming datasets if needed, 
        # but for local it's fine. For streaming, maybe limit?
        # Let's just run it.
        
        with torch.no_grad():
            # For streaming datasets, validation might be huge. Let's limit val batches if needed.
            # For now, assuming reasonable size or user uses --limit.
            # Actually, let's hardcode a limit for validation to ensure speed for baseline?
            # No, let's iterate.
            
            # Note: TarDataset is iterable, so we need to be careful not to exhaust it if it's infinite (it's not, it ends).
            # StreamingHFDataset is infinite? No, load_dataset(streaming=True) is iterable but finite if iterating over split?
            # Actually streaming datasets can be infinite if repeating. 
            # Our implementation just iterates `ds`.
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for i, batch in enumerate(val_pbar):
                if args.limit and i >= args.limit // 5: # Validate on smaller subset if limiting
                    break
                # If not limiting, we might want to break after some batches for HF to save time?
                # Let's verify 100 batches max for now to be safe for baseline.
                if i >= 100: 
                    break
                    
                l_input = batch['L'].to(args.device)
                ab_target = batch['ab'].to(args.device)
                
                ab_pred = model(l_input)
                loss = criterion(ab_pred, ab_target)
                
                val_loss += loss.item()
                val_count += 1
                val_pbar.set_postfix({'val_loss': val_loss / val_count})
                
        avg_val_loss = val_loss / val_count if val_count > 0 else 0
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            print(f"✅ Validation Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/baseline_best.pth")
            
        # Snapshot
        if epoch % 5 == 0 or epoch == 1:
            print("📸 Saving Snapshot...")
            # Re-create loader or just grab one? 
            # IterableDataset might be exhausted. 
            # We need a fresh iterator or handle it.
            # get_dataloader returns a new DataLoader which gets a new iterator.
            # But creating a new dataloader every time might be expensive?
            # Actually, we can just use the val_loader we have? 
            # IterableDataset: once iterated, it's done?
            # Yes, for TarDataset, __iter__ opens the file. So we can iterate again.
            save_snapshot(model, val_loader, args.device, epoch)
            
        # Save History
        with open("loss_history.json", "w") as f:
            json.dump(history, f)
            
    print("🎉 Training Complete!")

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="local_movienet", choices=['local_movienet', 'hf_imagenet', 'hf_places365'])
    parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of batches per epoch for debugging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    args.hf_token = os.getenv("HF_AUTH_TOKEN")
    
    train(args)
