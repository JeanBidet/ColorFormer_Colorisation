import torch
import sys
import os

# Pour pouvoir faire "from src.models.cffn import CFFN"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cffn import CFFN


def test_cffn_basic():
    print("🧪 Testing CFFN...")

    # Batch + spatial + canaux, même convention que LWMHSA : (B, H, W, C)
    B, H, W, C = 2, 32, 32, 96
    x = torch.randn(B, H, W, C)

    cffn = CFFN(dim=C)   # hidden_dim par défaut = 4*C si tu as mis ça

    y = cffn(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    # 1. On vérifie que la shape est conservée
    assert y.shape == x.shape, "❌ CFFN doit conserver la même shape (B, H, W, C)"

    # 2. On vérifie que le module est différentiable (backward OK)
    loss = y.mean()
    loss.backward()
    print("✅ Backprop OK, gradients calculés sans erreur.")


if __name__ == "__main__":
    test_cffn_basic()
