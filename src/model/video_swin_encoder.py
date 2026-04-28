"""
3D-SwinSTB encoder, built on torchvision's Video Swin Transformer.

torchvision provides `torchvision.models.video.SwinTransformer3d` which
implements Liu et al.'s Video Swin Transformer (the same architecture
referenced by Pan et al. in [13]). It supports configurable patch size,
window size, MLP ratio, embed dim, and depths — everything we need.

What this module does:
    - Instantiates SwinTransformer3d with Pan et al.'s hyperparameters.
    - Strips off the classification head (avgpool + linear).
    - Wraps the encoder so it returns the four feature maps the decoder
      needs:
          S1_en  — output of stage 1 (full patch-grid resolution)
          S2_en  — output of stage 2 (1/2 spatial)
          S3_en  — output of stage 3 (1/4 spatial, the encoder output)
          X_de   — output of the bottleneck (same resolution as S3_en)

Architecture (Pan et al. Section IV-A, with C=96):
    Input    : (B, 3, 20, 256, 256)
    Patch embed (2,4,4)          → (B, 10, 64, 64, 96)
    Swin stage 1 (depth 2, 4 heads) → S1_en (B, 10, 64, 64, 96)
    Patch Merging                 → (B, 10, 32, 32, 192)
    Swin stage 2 (depth 4, 8 heads) → S2_en (B, 10, 32, 32, 192)
    Patch Merging                 → (B, 10, 16, 16, 384)
    Swin stage 3 (depth 2, 16 heads)→ S3_en (B, 10, 16, 16, 384)
    Bottleneck (depth 2, 16 heads)  → X_de  (B, 10, 16, 16, 384)

Channel ordering note:
    torchvision's Video Swin uses *channels-last* (B, T, H, W, C) internally
    after the patch embedding, in contrast to most PyTorch convention
    (channels-first). We preserve this convention for the encoder outputs
    because that's what the Swin blocks expect. The decoder will permute
    as needed before its 3D Projection Layer at the very end.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models.video.swin_transformer import (
    PatchEmbed3d,
    PatchMerging,
    ShiftedWindowAttention3d,
    SwinTransformer3d,
)
from torchvision.models.swin_transformer import SwinTransformerBlock


# ─────────────────────────────────────────────────────────────────────────────
# Encoder + Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class VideoSwinEncoder(nn.Module):
    """
    3-stage Video Swin encoder + 2-block bottleneck.

    Args:
        in_channels: input image channels (3 for RGB).
        embed_dim: base feature dimension C (default 96).
        patch_size: 3D patch size (Tp, Hp, Wp). Default (2, 4, 4).
        window_size: 3D window size (P, M, M). Default (2, 7, 7).
        depths: blocks per encoder stage. Default (2, 4, 2) per Pan et al.
        num_heads: heads per encoder stage. Default (4, 8, 16) per Pan et al.
        mlp_ratio: MLP expansion ratio. Default 2.0 per Pan et al. eq. (14).
        bottleneck_depth: number of Swin blocks in the bottleneck. Default 2.
        stochastic_depth_prob: drop-path probability. Default 0.0
            (Pan et al. don't mention it; we leave it disabled to match.)

    Forward input:
        x: (B, in_channels, T, H, W) — channels-first PyTorch convention.

    Forward outputs (all channels-last for Swin block compatibility):
        s1_en: (B, T/Tp, H/Hp,  W/Wp,  C)
        s2_en: (B, T/Tp, H/2Hp, W/2Wp, 2C)
        s3_en: (B, T/Tp, H/4Hp, W/4Wp, 4C)
        x_de:  (B, T/Tp, H/4Hp, W/4Wp, 4C) — bottleneck output
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        window_size: Tuple[int, int, int] = (2, 7, 7),
        depths: Tuple[int, int, int] = (2, 4, 2),
        num_heads: Tuple[int, int, int] = (4, 8, 16),
        mlp_ratio: float = 2.0,
        bottleneck_depth: int = 2,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 3:
            raise ValueError(f"depths must have 3 stages, got {len(depths)}")
        if len(num_heads) != 3:
            raise ValueError(f"num_heads must have 3 stages, got {len(num_heads)}")
        if len(window_size) != 3 or len(patch_size) != 3:
            raise ValueError("window_size and patch_size must each have 3 elements")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = list(patch_size)
        self.window_size = list(window_size)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.mlp_ratio = mlp_ratio
        self.bottleneck_depth = bottleneck_depth

        # ----- Patch embedding -----
        # in_channels → embed_dim, with patch (Tp, Hp, Wp) and stride = patch
        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed3d(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        # ----- Encoder stages with patch-merging in between -----
        # Manually unroll: stage1 -> merge -> stage2 -> merge -> stage3
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        total_blocks = sum(depths) + bottleneck_depth
        block_id = 0

        for stage_idx, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage_dim = embed_dim * (2 ** stage_idx)  # 96, 192, 384
            stage_blocks = []
            for layer_idx in range(depth):
                # Stochastic depth per-block
                sd_prob = (
                    stochastic_depth_prob * float(block_id) / max(total_blocks - 1, 1)
                )
                # Shift size: even-indexed blocks use no shift, odd use half-window
                shift_size = [
                    0 if layer_idx % 2 == 0 else w // 2 for w in self.window_size
                ]
                stage_blocks.append(
                    SwinTransformerBlock(
                        dim=stage_dim,
                        num_heads=heads,
                        window_size=self.window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        dropout=0.0,
                        attention_dropout=0.0,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                block_id += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            # Add a PatchMerging after every stage except the last
            if stage_idx < len(depths) - 1:
                self.merges.append(PatchMerging(stage_dim, norm_layer))

        # ----- Bottleneck (extra Swin blocks at the same resolution as S3_en) -----
        bottleneck_dim = embed_dim * (2 ** (len(depths) - 1))  # 384
        bottleneck_blocks = []
        for layer_idx in range(bottleneck_depth):
            sd_prob = (
                stochastic_depth_prob * float(block_id) / max(total_blocks - 1, 1)
            )
            shift_size = [
                0 if layer_idx % 2 == 0 else w // 2 for w in self.window_size
            ]
            bottleneck_blocks.append(
                SwinTransformerBlock(
                    dim=bottleneck_dim,
                    num_heads=num_heads[-1],
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout=0.0,
                    attention_dropout=0.0,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                    attn_layer=ShiftedWindowAttention3d,
                )
            )
            block_id += 1
        self.bottleneck = nn.Sequential(*bottleneck_blocks)

        # Initialise weights similarly to torchvision's SwinTransformer3d
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            x: (B, C, T, H, W) — channels-first input.

        Returns:
            s1_en, s2_en, s3_en, x_de — all (B, T', H', W', C') channels-last.
        """
        # Patch embedding: (B, C, T, H, W) → (B, T', H', W', C')
        x = self.patch_embed(x)

        # Stage 1
        s1_en = self.stages[0](x)
        # Patch Merge → Stage 2
        x = self.merges[0](s1_en)
        s2_en = self.stages[1](x)
        # Patch Merge → Stage 3
        x = self.merges[1](s2_en)
        s3_en = self.stages[2](x)
        # Bottleneck at same resolution as s3_en
        x_de = self.bottleneck(s3_en)

        return s1_en, s2_en, s3_en, x_de