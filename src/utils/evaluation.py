import torch
import os
import matplotlib.pyplot as plt
import kornia
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.models.unet import ColorizationUNet
from src.utils.metrics import ColorizationEvaluator
from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE, NUM_WORKERS

def visualize_evaluation(l_input, ab_pred, ab_target, batch_idx, output_dir="outputs/evaluation_viz"):
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
    axes[1].set_title("Prediction")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_target)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"eval_batch_{batch_idx}.png"))
    plt.close()

def evaluate_baseline(source, tar_path=None, hf_token=None, device='cpu', num_batches=10, checkpoint=None):
    print(f"🚀 Starting Baseline Evaluation on {source}...")
    
    # 1. Setup Data
    loader = get_dataloader(
        source=source, 
        split='val',
        tar_path=tar_path, 
        hf_token=hf_token, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    # 2. Setup Model
    model = ColorizationUNet().to(device)
    
    if checkpoint and os.path.exists(checkpoint):
        print(f"📥 Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        # Handle both full checkpoint dict and direct state_dict
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)
    else:
        print("⚠️ No checkpoint found or provided. Using random weights (Untrained Baseline).")
        
    model.eval()
    
    # 3. Setup Evaluator
    evaluator = ColorizationEvaluator(device=device)
    
    results = {
        'psnr_ab': [],
        'ssim': [],
        'lpips': [],
        'brisque': []
    }
    
    print(f"🔄 Evaluating {num_batches} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=num_batches)):
            if i >= num_batches:
                break
                
            l_input = batch['L'].to(device)   # (B, 1, H, W)
            ab_target = batch['ab'].to(device) # (B, 2, H, W)
            
            # Forward Pass
            ab_pred = model(l_input)
            
            # Grayscale Baseline (Predict 0 for ab)
            ab_zeros = torch.zeros_like(ab_pred)
            
            # Visualize first batch
            if i == 0:
                visualize_evaluation(l_input, ab_pred, ab_target, i)
            
            # Compute Metrics for Model
            batch_metrics = evaluator.evaluate_batch(l_input, ab_pred, ab_target)
            for k, v in batch_metrics.items():
                results[k].append(v)
                
            # Compute Metrics for Grayscale Baseline
            gray_metrics = evaluator.evaluate_batch(l_input, ab_zeros, ab_target)
            for k, v in gray_metrics.items():
                if f"gray_{k}" not in results:
                    results[f"gray_{k}"] = []
                results[f"gray_{k}"].append(v)
                
    # 4. Aggregation
    avg_results = {k: sum(v)/len(v) if len(v) > 0 else 0.0 for k, v in results.items()}
    
    # 5. Output
    print("\n📊 Baseline Evaluation Results")
    
    # Format as two rows: Model vs Grayscale
    model_row = {'Model': 'U-Net'}
    gray_row = {'Model': 'Grayscale'}
    
    for k in ['psnr_ab', 'ssim', 'lpips', 'brisque']:
        model_row[k] = avg_results.get(k, 0.0)
        gray_row[k] = avg_results.get(f"gray_{k}", 0.0)
        
    df = pd.DataFrame([model_row, gray_row])
    print(df.to_markdown(index=False))
    
    return avg_results
