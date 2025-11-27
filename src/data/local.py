import tarfile
import io
import torch
from torch.utils.data import IterableDataset
from PIL import Image

class TarDataset(IterableDataset):
    """
    Streams images from a local .tar archive.
    """
    def __init__(self, tar_path, preprocessor, split='train', limit=None):
        super(TarDataset, self).__init__()
        self.tar_path = tar_path
        self.preprocessor = preprocessor
        self.split = split
        self.limit = limit

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        count = 0
        try:
            with tarfile.open(self.tar_path, 'r|*') as tar:
                for i, member in enumerate(tar):
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue
                    
                    # Deterministic Splitting
                    # Use hash of filename to decide split
                    # 90% Train, 10% Val
                    h = int(hash(member.name))
                    is_val = (h % 10) == 9 # 10% chance
                    
                    if self.split == 'train' and is_val:
                        continue
                    if self.split == 'val' and not is_val:
                        continue

                    # Simple worker splitting
                    if worker_info is not None:
                        if i % worker_info.num_workers != worker_info.id:
                            continue

                    try:
                        f = tar.extractfile(member)
                        if f is None: continue
                        
                        image = Image.open(io.BytesIO(f.read()))
                        data = self.preprocessor(image)
                        data = self.preprocessor(image)
                        data['id'] = member.name
                        yield data
                        
                        count += 1
                        if self.limit is not None and count >= self.limit:
                            return
                        
                    except Exception as e:
                        # print(f"Error loading {member.name}: {e}")
                        continue
        except Exception as e:
            print(f"Error opening tar {self.tar_path}: {e}")
            raise e
