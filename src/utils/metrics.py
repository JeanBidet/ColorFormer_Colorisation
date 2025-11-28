import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# Note: BRISQUE might require 'piq' or newer torchmetrics. 
# We will try to import from torchmetrics, if not available we handle it.
try:
    from torchmetrics.image import BRISQUEScore
    USE_PIQ = False
except ImportError:
    try:
        from piq import brisque as piq_brisque
        USE_PIQ = True
        BRISQUEScore = True # Flag to enable it
    except ImportError:
        BRISQUEScore = None
        USE_PIQ = False

import kornia

class ColorizationEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Standard Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # BRISQUE
        if BRISQUEScore:
            if not USE_PIQ:
                self.brisque = BRISQUEScore().to(device)
            else:
                self.brisque = "piq" # Marker
        else:
            print("Warning: BRISQUEScore not found in torchmetrics and piq not installed. BRISQUE will be skipped.")
            self.brisque = None

    def compute_psnr_ab(self, preds_ab, target_ab):
        """
        Computes PSNR strictly on the ab channels.
        Data Range is 2.0 because values are in [-1, 1].
        """
        mse = torch.mean((preds_ab - target_ab) ** 2)
        if mse == 0:
            return float('inf')
        
        data_range = 2.0
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        return psnr.item()

    def evaluate_batch(self, l_input, ab_pred, ab_target):
        """
        Evaluates a batch of predictions.
        
        Args:
            l_input: (B, 1, H, W) in [-1, 1]
            ab_pred: (B, 2, H, W) in [-1, 1]
            ab_target: (B, 2, H, W) in [-1, 1]
            
        Returns:
            dict: { 'ssim': float, 'lpips': float, 'psnr_ab': float, 'brisque': float }
        """
        metrics = {}
        
        # 1. PSNR_ab (Raw Tensors)
        metrics['psnr_ab'] = self.compute_psnr_ab(ab_pred, ab_target)
        
        # 2. Convert to RGB for Perceptual Metrics
        # Denormalize L and ab to standard Lab ranges for Kornia conversion?
        # Kornia lab_to_rgb expects: L [0, 100], a [-128, 127], b [-128, 127]
        # Our tensors are [-1, 1].
        
        def denorm_lab(l, ab):
            l_denorm = (l + 1.0) * 50.0
            ab_denorm = ab * 128.0
            return torch.cat([l_denorm, ab_denorm], dim=1)
            
        lab_pred = denorm_lab(l_input, ab_pred)
        lab_target = denorm_lab(l_input, ab_target)
        
        rgb_pred = kornia.color.lab_to_rgb(lab_pred)
        rgb_target = kornia.color.lab_to_rgb(lab_target)
        
        # Clip to [0, 1] for metrics
        rgb_pred = torch.clamp(rgb_pred, 0, 1)
        rgb_target = torch.clamp(rgb_target, 0, 1)
        
        # 3. SSIM
        metrics['ssim'] = self.ssim(rgb_pred, rgb_target).item()
        
        # 4. LPIPS
        # LPIPS expects input in range [-1, 1] usually for its internal net, 
        # but torchmetrics implementation handles normalization if configured?
        # Torchmetrics LPIPS expects [0, 1] by default and normalizes internally if normalize=True (default False).
        # Actually, let's check docs. Usually expects [0, 1] or [-1, 1]. 
        # We will pass [0, 1] and let it handle it (it scales to [-1, 1] internally usually).
        metrics['lpips'] = self.lpips(rgb_pred, rgb_target).item()
        
        # 5. BRISQUE (No-Reference)
        if self.brisque:
            try:
                # BRISQUE expects [0, 1]
                
                if self.brisque == "piq":
                    # piq.brisque expects (N, C, H, W) in [0, 1]
                    # Returns a tensor of size (N,)
                    # We want the mean.
                    # Ensure input is [0, 1]
                    score = piq_brisque(rgb_pred, data_range=1.0).mean().item()
                else:
                    # Torchmetrics path
                    score = self.brisque(rgb_pred).mean().item()
                
                if score == 0.0 or score != score: # Check for 0.0 or NaN
                     # print("⚠️ BRISQUE Computation Failed (Result 0.0 or NaN)")
                     metrics['brisque'] = float('nan')
                else:
                    metrics['brisque'] = score
                    
            except Exception as e:
                # print(f"BRISQUE error: {e}")
                metrics['brisque'] = float('nan')
        else:
            metrics['brisque'] = float('nan')
            
        return metrics
