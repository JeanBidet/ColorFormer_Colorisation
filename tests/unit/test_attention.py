import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.attention import LWMHSA

def test_lwmhsa():
    print("🧪 Testing LWMHSA...")
    
    B, H, W, C = 2, 64, 64, 96
    x = torch.randn(B, H, W, C)
    
    # Test 1: No Shift
    print("\n[Test 1] Shift Size = 0")
    lwmhsa = LWMHSA(dim=C, num_heads=4, window_size=7, shift_size=0)
    out = lwmhsa(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # Test 2: Shifted Window
    print("\n[Test 2] Shift Size = 3")
    lwmhsa_shifted = LWMHSA(dim=C, num_heads=4, window_size=7, shift_size=3)
    out_shifted = lwmhsa_shifted(x)
    print(f"Input: {x.shape}, Output: {out_shifted.shape}")
    assert out_shifted.shape == x.shape, "Shape mismatch!"
    
    # Test 3: Padding (Odd dimensions)
    print("\n[Test 3] Padding (60x60 input, window 7)")
    H_odd, W_odd = 60, 60
    x_odd = torch.randn(B, H_odd, W_odd, C)
    out_odd = lwmhsa_shifted(x_odd)
    print(f"Input: {x_odd.shape}, Output: {out_odd.shape}")
    assert out_odd.shape == x_odd.shape, "Shape mismatch after padding!"
    
    print("\n✅ All LWMHSA tests passed.")

if __name__ == "__main__":
    test_lwmhsa()
