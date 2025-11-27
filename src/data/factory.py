from torch.utils.data import DataLoader
from .core import CinematicPreprocessor
from .local import TarDataset
from .remote import StreamingHFDataset
from configs.config import HF_IMAGENET_ID, HF_PLACES365_ID

def get_dataloader(source, tar_path=None, hf_token=None, batch_size=32, num_workers=4, image_size=(256, 256)):
    """
    Factory for DataLoaders.
    
    Args:
        source (str): 'local_movienet', 'hf_imagenet', 'hf_places365'
        tar_path (str): Path for local tar.
        hf_token (str): Token for gated HF datasets.
    """
    preprocessor = CinematicPreprocessor(image_size=image_size)
    
    if source == 'local_movienet':
        if not tar_path:
            raise ValueError("tar_path is required for local_movienet")
        dataset = TarDataset(tar_path, preprocessor)
        
    elif source == 'hf_imagenet':
        dataset = StreamingHFDataset(
            dataset_name=HF_IMAGENET_ID, 
            split="train", 
            preprocessor=preprocessor, 
            auth_token=hf_token
        )
        
    elif source == 'hf_places365':
        dataset = StreamingHFDataset(
            dataset_name=HF_PLACES365_ID, 
            split="train", 
            preprocessor=preprocessor, 
            auth_token=hf_token
        )
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )
