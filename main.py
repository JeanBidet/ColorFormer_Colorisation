import argparse
import os
from dotenv import load_dotenv
from src.data import get_dataloader
from src.utils import visualize_batch, create_dummy_tar
from configs.config import DEFAULT_TAR_PATH, MOVIENET_PATH, BATCH_SIZE, NUM_WORKERS

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Visualize Colorization Pipeline")
    parser.add_argument("--source", type=str, default="local_movienet", choices=['local_movienet', 'hf_imagenet', 'hf_places365'], help="Data source")
    parser.add_argument("--tar_path", type=str, default="dataset/MovieNet_valid/valid_mv_raw/tt0032138.tar", help="Path to the .tar dataset (for local_movienet)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Token (for hf_imagenet)")
    args = parser.parse_args()
    
    # Define sources to visualize
    sources = [
        ('local_movienet', args.tar_path),
        ('hf_imagenet', None),
        ('hf_places365', None)
    ]
    
    hf_token = args.hf_token
    
    for source_name, tar_path in sources:
        print(f"\n--- Visualizing {source_name} ---")
        try:
            # Handle local tar path logic
            current_tar = tar_path if source_name == 'local_movienet' else None
            
            if source_name == 'local_movienet':
                # Auto-detect tar if default/dummy and directory exists
                if current_tar == "dummy_dataset.tar":
                    default_dir = "dataset/MovieNet_valid/valid_mv_raw/"
                    if os.path.exists(default_dir):
                        tars = [f for f in os.listdir(default_dir) if f.endswith('.tar')]
                        if tars:
                            current_tar = os.path.join(default_dir, tars[0])
                            print(f"Auto-detected MovieNet tar: {current_tar}")
                
                if current_tar == "dummy_dataset.tar" and not os.path.exists(current_tar):
                    create_dummy_tar(current_tar)
                if not os.path.exists(current_tar):
                    print(f"Skipping {source_name}: File {current_tar} not found.")
                    continue

            loader = get_dataloader(
                source=source_name, 
                tar_path=current_tar, 
                hf_token=hf_token,
                batch_size=4, 
                num_workers=0 
            )
            
            # Custom visualize call to save with specific name
            output_filename = f"test_vizualisation/viz_{source_name}.png"
            
            # Fetch batch manually to pass to viz function if needed, 
            # but visualize_batch takes loader. 
            # Let's modify visualize_batch to accept filename or just rename after.
            # Actually, better to modify visualize_batch to take output_path.
            
            # We will inline the viz logic or modify the function. 
            # Let's modify the function signature in a separate edit or just monkey-patch/copy-paste logic?
            # Cleaner to modify the function signature first.
            # But I can't do two edits to the same file in one turn easily if they overlap or if I want to be safe.
            # I'll just update the loop to call a modified version or set a global/argument.
            
            # Wait, I can't easily change the function signature AND the main block in one `replace_file_content` if they are far apart.
            # `visualize_batch` is at line 33. Main is at 108.
            # I will use `multi_replace_file_content` if I had it, but I have `replace_file_content`.
            # I will just update the main block to rename the file after generation.
            
            visualize_batch(loader)
            
            if os.path.exists("visualization_output.png"):
                if os.path.exists(output_filename):
                    os.remove(output_filename)
                os.rename("visualization_output.png", output_filename)
                print(f"Saved to {output_filename}")
            
        except Exception as e:
            print(f"Failed to visualize {source_name}: {e}")

if __name__ == "__main__":
    main()
