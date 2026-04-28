"""
3D Patch Expanding layer — the inverse of torchvision's PatchMerging.

Implements Pan et al. equation (12).

Mechanics (for "double spatial, halve channels"):
    Input:  (B, T, H, W, C)
    Step 1: linear C → 2C    (a learned linear projection)
    Step 2: rearrange       (redistribute channels into a 2x2 spatial grid)
            (B, T, H, W, 2C) → (B, T, 2H, 2W, C/2)

The two-step expand-then-rearrange shuffles channels into spatial positions
without losing information. It's the natural inverse of PatchMerging,
which concatenates a 2x2 spatial neighbourhood (4C) and projects to 2C.

Note: we only do *spatial* expansion (H and W double). The time axis T
is left untouched — same as how PatchMerging only halves H and W.
That's because Pan et al.'s patch size has a different temporal vs.
spatial factor, and time stays at T/Tp throughout the encoder/decoder.
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchExpanding3d(nn.Module):
    """
    Spatial 2x upsample + channel halving for 5D channels-last tensors.

    Args:
        dim: input channel dimension C.
        norm_layer: normalization applied after the rearrange. Default LayerNorm.

    Forward:
        Input  shape: (B, T, H, W, C)
        Output shape: (B, T, 2H, 2W, C // 2)
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for PatchExpanding3d, got {dim}")
        self.dim = dim
        # Step 1: linear C → 2C (no bias matches Swin-UNet convention)
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        # Step 2: rearrange will produce (B, T, 2H, 2W, C/2). Norm at output dim.
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        x = self.expand(x)  # (B, T, H, W, 2C)
        # Rearrange: split the 2C channels into a 2x2 spatial pattern with C/2 each.
        # einops pattern: 'b t h w (p1 p2 c) -> b t (h p1) (w p2) c'
        # with p1=p2=2 means each 2C-vector unfolds to a 2x2 block of C/2 vectors.
        x = rearrange(x, 'b t h w (p1 p2 c) -> b t (h p1) (w p2) c', p1=2, p2=2)
        # x: (B, T, 2H, 2W, C/2)
        x = self.norm(x)
        return x