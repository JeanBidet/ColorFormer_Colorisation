import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from src.models.unet import ColorizationUNet
from src.utils.metrics import ColorizationEvaluator
from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE, NUM_WORKERS, DEFAULT_TAR_PATH, MOVIENET_PATH

def evaluate_baseline(source, tar_path=None, hf_token=None, device='cpu', num_batches=10):
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
    
    # 2. Setup Model (Untrained Baseline)
    model = ColorizationUNet().to(device)
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
            
            # Compute Metrics for Model
            batch_metrics = evaluator.evaluate_batch(l_input, ab_pred, ab_target)
            for k, v in batch_metrics.items():
                results[k].append(v)
                
            # Compute Metrics for Grayscale Baseline
            gray_metrics = evaluator.evaluate_batch(l_input, ab_zeros, ab_target)
            for k, v in gray_metrics.items():
                # Store with 'gray_' prefix or separate dict? 
                # Let's use a separate dict key in results for simplicity if we want to average later.
                # Or just print it?
                # Let's append to a new list in results
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

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="local_movienet", choices=['local_movienet', 'hf_imagenet', 'hf_places365'])
    parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Auto-detect tar logic (copied from main.py)
    current_tar = args.tar_path
    if args.source == 'local_movienet':
        if current_tar == DEFAULT_TAR_PATH:
            if os.path.exists(MOVIENET_PATH):
                tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
                if tars:
                    current_tar = os.path.join(MOVIENET_PATH, tars[0])
    
    hf_token = os.getenv("HF_AUTH_TOKEN")
    
    evaluate_baseline(
        source=args.source,
        tar_path=current_tar,
        hf_token=hf_token,
        device=args.device,
        num_batches=args.num_batches
    )
