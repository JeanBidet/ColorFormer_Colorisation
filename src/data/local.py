import tarfile
import io
import torch
from torch.utils.data import IterableDataset
from PIL import Image

class TarDataset(IterableDataset):
    """
    Streams images from a local .tar archive.
    """
    def __init__(self, tar_path, preprocessor):
        super(TarDataset, self).__init__()
        self.tar_path = tar_path
        self.preprocessor = preprocessor

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        try:
            with tarfile.open(self.tar_path, 'r|*') as tar:
                for i, member in enumerate(tar):
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue

                    # Simple worker splitting
                    if worker_info is not None:
                        if i % worker_info.num_workers != worker_info.id:
                            continue

                    try:
                        f = tar.extractfile(member)
                        if f is None: continue
                        
                        image = Image.open(io.BytesIO(f.read()))
                        yield self.preprocessor(image)
                        
                    except Exception as e:
                        # print(f"Error loading {member.name}: {e}")
                        continue
        except Exception as e:
            print(f"Error opening tar {self.tar_path}: {e}")
            raise e
