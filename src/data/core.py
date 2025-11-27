from torchvision import transforms
import kornia
import torch

class CinematicPreprocessor:
    """
    Common Core Preprocessing Logic.
    Ensures identical transformations for both Local and Remote streams.
    """
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        
        # 1. PIL -> Tensor (0-1) + Augmentations
        self.pre_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Converts to [0, 1]
        ])

    def __call__(self, image):
        """
        Args:
            image (PIL.Image): Input RGB image.
        Returns:
            dict: {'L': tensor, 'ab': tensor, 'original': tensor}
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 1. Apply Pre-transforms
        img_tensor = self.pre_transforms(image) # (C, H, W)
        
        # Add batch dim for Kornia
        img_tensor_b = img_tensor.unsqueeze(0) 

        # 2. Kornia Conversion: RGB -> Lab
        lab_tensor = kornia.color.rgb_to_lab(img_tensor_b)
        
        # 3. Normalization to [-1, 1]
        # L: [0, 100] -> [-1, 1]
        l_channel = lab_tensor[:, 0:1, :, :]
        l_norm = (l_channel / 50.0) - 1.0
        
        # ab: [-128, 127] -> [-1, 1]
        ab_channel = lab_tensor[:, 1:3, :, :]
        ab_norm = ab_channel / 128.0
        
        return {
            'L': l_norm.squeeze(0), 
            'ab': ab_norm.squeeze(0), 
            'original': img_tensor
        }
