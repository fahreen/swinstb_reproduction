"""
3D-SwinSTB decoder — U-Net-style upsampling path.

Implements Pan et al.'s predictor D_β3D (Section IV-A, equation (10)).

Pipeline at each decoder stage:
    1. Concatenate the previous decoder output with the corresponding
       encoder skip connection along the channel axis.
    2. Apply a 1x1x1 conv (= linear projection) to bring the channel
       count back to the "natural" size for that resolution. This is
       the "channel cleanup" step that Pan et al.'s equations gloss over;
       we add it explicitly for clean shape arithmetic, matching
       Swin-UNet (Cao et al., 2021) practice.
    3. Run Swin Transformer blocks at this resolution.
    4. Apply Patch Expanding to upsample to the next resolution
       (except after the final stage — that's the projection layer's job).

Stage-by-stage (with C=96):
    Input from bottleneck:  (B, 10, 16, 16, 384)
    Stage 1 — concat with S3_en (B, 10, 16, 16, 384) → cat 768
              cleanup proj 768→384, Swin x 2 (heads=16)
              → (B, 10, 16, 16, 384)
    PatchExp → (B, 10, 32, 32, 192)
    Stage 2 — concat with S2_en (B, 10, 32, 32, 192) → cat 384
              cleanup proj 384→192, Swin x 4 (heads=8)
              → (B, 10, 32, 32, 192)
    PatchExp → (B, 10, 64, 64, 96)
    Stage 3 — concat with S1_en (B, 10, 64, 64, 96) → cat 192
              cleanup proj 192→96, Swin x 2 (heads=4)
              → (B, 10, 64, 64, 96)

The (B, 10, 64, 64, 96) output is what feeds into the 3D Projection Layer,
which converts back to pixel resolution (B, 3, 20, 256, 256).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models.video.swin_transformer import ShiftedWindowAttention3d
from torchvision.models.swin_transformer import SwinTransformerBlock

from .patch_expanding import PatchExpanding3d


class CleanupProjection(nn.Module):
    """
    1x1x1 conv that operates on a channels-last 5D tensor.

    Used to bring the post-concat channel count back to the natural size.
    Implementing as Linear (rather than Conv3d on permuted axes) is
    cheaper and equivalent at kernel size 1.

    Args:
        in_dim: channel dim before reduction.
        out_dim: channel dim after reduction (typically in_dim // 2).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


def _build_swin_stage(
    dim: int,
    num_heads: int,
    depth: int,
    window_size: List[int],
    mlp_ratio: float,
) -> nn.Sequential:
    """
    Build a Sequential of `depth` SwinTransformerBlock layers,
    alternating between window and shifted-window attention.
    """
    blocks = []
    for layer_idx in range(depth):
        shift_size = [
            0 if layer_idx % 2 == 0 else w // 2 for w in window_size
        ]
        blocks.append(
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
                attention_dropout=0.0,
                stochastic_depth_prob=0.0,
                norm_layer=nn.LayerNorm,
                attn_layer=ShiftedWindowAttention3d,
            )
        )
    return nn.Sequential(*blocks)


class Decoder3D(nn.Module):
    """
    Three-stage U-Net decoder using Swin Transformer blocks.

    Args:
        embed_dim: base channel dim C (default 96).
        depths: blocks per decoder stage. Default (2, 4, 2) — same shape as
            the encoder, matching Pan et al.'s symmetric design.
        num_heads: heads per decoder stage. Default (16, 8, 4) — reverse of
            the encoder's (4, 8, 16), per Pan et al.
        window_size: 3D window size (P, M, M). Default (2, 7, 7).
        mlp_ratio: MLP expansion ratio. Default 2.0.

    Forward inputs (all channels-last):
        x_de:  (B, T, H,  W,  4C)  — bottleneck output
        s3_en: (B, T, H,  W,  4C)  — encoder skip 3
        s2_en: (B, T, 2H, 2W, 2C)  — encoder skip 2
        s1_en: (B, T, 4H, 4W, C)   — encoder skip 1

    Forward output:
        (B, T, 4H, 4W, C) — feature tensor at patch-grid resolution.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depths: Tuple[int, int, int] = (2, 4, 2),
        num_heads: Tuple[int, int, int] = (16, 8, 4),
        window_size: Tuple[int, int, int] = (2, 7, 7),
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if len(depths) != 3:
            raise ValueError(f"depths must have 3 stages, got {len(depths)}")
        if len(num_heads) != 3:
            raise ValueError(f"num_heads must have 3 stages, got {len(num_heads)}")

        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = list(window_size)
        self.mlp_ratio = mlp_ratio

        # Channel dimensions per stage:
        #   Stage 1 operates at 4C  (matches s3_en)
        #   Stage 2 operates at 2C  (matches s2_en)
        #   Stage 3 operates at  C  (matches s1_en)
        c1 = embed_dim * 4
        c2 = embed_dim * 2
        c3 = embed_dim

        # ----- Stage 1: at 4C resolution -----
        # After concat(x_de, s3_en): 8C. Cleanup back to 4C.
        self.cleanup1 = CleanupProjection(in_dim=2 * c1, out_dim=c1)
        self.stage1 = _build_swin_stage(
            dim=c1, num_heads=num_heads[0],
            depth=depths[0], window_size=self.window_size, mlp_ratio=mlp_ratio,
        )
        # PatchExpand: 4C → 2C, spatial 2x
        self.expand1 = PatchExpanding3d(dim=c1)

        # ----- Stage 2: at 2C resolution -----
        # After concat(expand1_out, s2_en): 4C. Cleanup back to 2C.
        self.cleanup2 = CleanupProjection(in_dim=2 * c2, out_dim=c2)
        self.stage2 = _build_swin_stage(
            dim=c2, num_heads=num_heads[1],
            depth=depths[1], window_size=self.window_size, mlp_ratio=mlp_ratio,
        )
        # PatchExpand: 2C → C, spatial 2x
        self.expand2 = PatchExpanding3d(dim=c2)

        # ----- Stage 3: at C resolution -----
        # After concat(expand2_out, s1_en): 2C. Cleanup back to C.
        self.cleanup3 = CleanupProjection(in_dim=2 * c3, out_dim=c3)
        self.stage3 = _build_swin_stage(
            dim=c3, num_heads=num_heads[2],
            depth=depths[2], window_size=self.window_size, mlp_ratio=mlp_ratio,
        )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_de: torch.Tensor,
        s3_en: torch.Tensor,
        s2_en: torch.Tensor,
        s1_en: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_de:  bottleneck output, (B, T, H, W, 4C)
            s3_en: encoder stage-3 output, (B, T, H, W, 4C)
            s2_en: encoder stage-2 output, (B, T, 2H, 2W, 2C)
            s1_en: encoder stage-1 output, (B, T, 4H, 4W, C)

        Returns:
            Tensor of shape (B, T, 4H, 4W, C) — features at patch-grid resolution.
        """
        # Stage 1: concat with s3_en, cleanup, Swin blocks, expand
        x = torch.cat([x_de, s3_en], dim=-1)   # (B, T, H, W, 8C)
        x = self.cleanup1(x)                    # (B, T, H, W, 4C)
        x = self.stage1(x)                      # (B, T, H, W, 4C)
        x = self.expand1(x)                     # (B, T, 2H, 2W, 2C)

        # Stage 2: concat with s2_en, cleanup, Swin blocks, expand
        x = torch.cat([x, s2_en], dim=-1)       # (B, T, 2H, 2W, 4C)
        x = self.cleanup2(x)                    # (B, T, 2H, 2W, 2C)
        x = self.stage2(x)                      # (B, T, 2H, 2W, 2C)
        x = self.expand2(x)                     # (B, T, 4H, 4W, C)

        # Stage 3: concat with s1_en, cleanup, Swin blocks (no further expand)
        x = torch.cat([x, s1_en], dim=-1)       # (B, T, 4H, 4W, 2C)
        x = self.cleanup3(x)                    # (B, T, 4H, 4W, C)
        x = self.stage3(x)                      # (B, T, 4H, 4W, C)

        return x