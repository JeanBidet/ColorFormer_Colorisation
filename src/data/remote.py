import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

class StreamingHFDataset(IterableDataset):
    """
    Streams images from Hugging Face Datasets.
    """
    def __init__(self, dataset_name, split, preprocessor, auth_token=None):
        super(StreamingHFDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.preprocessor = preprocessor
        self.auth_token = auth_token

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name, 
            split=self.split, 
            streaming=True, 
            token=self.auth_token
        )
        
        # Sharding for workers would go here if needed
        
        try:
            for sample in ds:
                try:
                    # HF datasets usually have 'image' key
                    if 'image' not in sample:
                        continue
                        
                    image = sample['image']
                    
                    # Filter non-RGB (e.g. Grayscale, RGBA)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    yield self.preprocessor(image)
                    
                except Exception as e:
                    # print(f"Error processing HF sample: {e}")
                    continue
        except Exception as e:
            print(f"Error streaming from HF {self.dataset_name}: {e}")
            raise e
