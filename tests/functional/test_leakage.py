import sys
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.factory import get_dataloader
from configs.config import BATCH_SIZE, NUM_WORKERS, DEFAULT_TAR_PATH, MOVIENET_PATH

def audit_data_leakage(source, tar_path=None, hf_token=None, limit=None):
    print(f"🕵️ Starting Data Leakage Audit on {source}...")
    
    # 1. Load Train Loader
    print("Loading Train Split...")
    train_loader = get_dataloader(
        source=source, 
        split='train',
        tar_path=tar_path, 
        hf_token=hf_token, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    # 2. Load Val Loader
    print("Loading Val Split...")
    val_loader = get_dataloader(
        source=source, 
        split='val',
        tar_path=tar_path, 
        hf_token=hf_token, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    train_ids = set()
    val_ids = set()
    
    print("Collecting Train IDs...")
    try:
        for i, batch in enumerate(tqdm(train_loader)):
            if limit and i >= limit:
                break
            for img_id in batch['id']:
                train_ids.add(img_id)
    except Exception as e:
        print(f"Error in Train Loop: {e}")
        
    print(f"Collected {len(train_ids)} Train IDs.")
    
    print("Collecting Val IDs...")
    try:
        for i, batch in enumerate(tqdm(val_loader)):
            if limit and i >= limit:
                break
            for img_id in batch['id']:
                val_ids.add(img_id)
    except Exception as e:
        print(f"Error in Val Loop: {e}")
        
    print(f"Collected {len(val_ids)} Val IDs.")
    
    # 3. Check Intersection
    overlap = train_ids.intersection(val_ids)
    num_overlap = len(overlap)
    
    print("\n🚨 ANALYSIS:")
    if num_overlap > 0:
        print(f"❌ DATA LEAKAGE DETECTED! Found {num_overlap} overlapping samples.")
        print(f"Examples: {list(overlap)[:5]}")
    else:
        print("✅ NO LEAKAGE DETECTED. Train and Val sets are disjoint.")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="local_movienet")
    parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    
    hf_token = os.getenv("HF_AUTH_TOKEN")
    
    # Auto-detect tar
    current_tar = args.tar_path
    if args.source == 'local_movienet':
        if current_tar == DEFAULT_TAR_PATH:
            if os.path.exists(MOVIENET_PATH):
                tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
                if tars:
                    current_tar = os.path.join(MOVIENET_PATH, tars[0])

    audit_data_leakage(args.source, current_tar, hf_token, args.limit)
