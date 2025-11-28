import argparse
import os
import sys
from dotenv import load_dotenv
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer import train
from src.utils.evaluation import evaluate_baseline
from configs.config import DEFAULT_TAR_PATH, MOVIENET_PATH, NUM_WORKERS

def run_tests(args):
    print("🧪 Running Tests...")
    # Simple runner for now, can be replaced by pytest
    test_modules = [
        "tests.functional.test_pipeline",
        "tests.functional.test_leakage"
    ]
    
    for module in test_modules:
        print(f"\n--- Running {module} ---")
        exit_code = os.system(f"python -m {module}")
        if exit_code != 0:
            print(f"❌ Test {module} failed with exit code {exit_code}")
            sys.exit(exit_code)
    print("\n✅ All tests passed.")

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Cinematic Colorizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # --- TRAIN ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--source", type=str, default="local_movienet", choices=['local_movienet', 'hf_imagenet', 'hf_places365'])
    train_parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    train_parser.add_argument("--limit", type=int, default=20000, help="Limit dataset size")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- EVALUATE ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--source", type=str, default="local_movienet", choices=['local_movienet', 'hf_imagenet', 'hf_places365'])
    eval_parser.add_argument("--tar_path", type=str, default=DEFAULT_TAR_PATH)
    eval_parser.add_argument("--num_batches", type=int, default=10)
    eval_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_unet_best.pth")
    
    # --- TEST ---
    test_parser = subparsers.add_parser("test", help="Run tests")
    
    args = parser.parse_args()
    
    # Common Setup
    args.hf_token = os.getenv("HF_AUTH_TOKEN")
    
    # Auto-detect tar logic
    if hasattr(args, 'source') and args.source == 'local_movienet' and hasattr(args, 'tar_path') and args.tar_path == DEFAULT_TAR_PATH:
        if os.path.exists(MOVIENET_PATH):
            tars = [f for f in os.listdir(MOVIENET_PATH) if f.endswith('.tar')]
            if tars:
                args.tar_path = os.path.join(MOVIENET_PATH, tars[0])

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_baseline(
            source=args.source,
            tar_path=args.tar_path,
            hf_token=args.hf_token,
            device=args.device,
            num_batches=args.num_batches,
            checkpoint=args.checkpoint
        )
    elif args.command == "test":
        run_tests(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
