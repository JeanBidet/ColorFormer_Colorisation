import torch
import torch.nn as nn

class CFFN(nn.Module):
    """
    Color Feed-Forward Network (CFFN) conforme au papier ColorFormer.
    Linear -> Reshape -> Depthwise Conv -> GELU -> Flatten -> Linear
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)

        # Depthwise Convolution
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim  # = depthwise
        )

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Linear
        x = self.fc1(x)  # (B, H, W, hidden)

        # Reshape for convolution
        x = x.permute(0, 3, 1, 2)  # (B, hidden, H, W)

        # Depthwise Convolution
        x = self.dwconv(x)
        x = self.act(x)

        # Back to (B, H, W, hidden)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Final Linear
        x = self.fc2(x)  # (B, H, W, C)

        return x
