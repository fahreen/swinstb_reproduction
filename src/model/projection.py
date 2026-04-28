"""
3D Projection Layer — final output stage of 3D-SwinSTB.

Implements Pan et al. equation (21):

    S³_Tr  ──3DConv⁻¹─────►  S³_Tr'  ──Rⁿ(3DConv⁻¹)─────►  X̂*_{T+1:T+K}
           (Tp,Hp,Wp)                  (1,1,1)
           stride=(Tp,Hp,Wp)           repeated n times

Two stages:
    1. Main upsampler: a single transposed 3D conv with kernel and stride
       both equal to the patch size (Tp, Hp, Wp) = (2, 4, 4). This inverts
       the patch embedding from the encoder, taking the patch-grid
       feature tensor back to pixel resolution.

       (B, C, T/Tp, H/Hp, W/Wp)   →   (B, C, T, H, W)

    2. Repeated channel reduction: n = ⌈log2(C/3)⌉ stacked 1x1x1
       transposed convs, each halving the channel count, until reaching 3.

       For C=96, n=5:  96 → 48 → 24 → 12 → 6 → 3

Input/output convention:
    The decoder produces channels-LAST tensors (B, T, H, W, C) because that's
    what Swin Transformer blocks operate on. But Conv3d / ConvTranspose3d
    expect channels-FIRST (B, C, T, H, W). We permute on entry and accept
    a channels-last input here, returning channels-first at the output.
"""

import math

import torch
import torch.nn as nn


class ProjectionLayer3D(nn.Module):
    """
    Final 3D inverse-conv pyramid that maps decoder features → RGB pixel video.

    Args:
        in_channels: feature dim C from the decoder (default 96).
        out_channels: output channels (3 for RGB).
        patch_size: (Tp, Hp, Wp) — must match the encoder's patch size.
            Default (2, 4, 4).

    Forward input (channels-last):
        x: (B, T/Tp, H/Hp, W/Wp, C)

    Forward output (channels-first):
        (B, out_channels, T, H, W)
    """

    def __init__(
        self,
        in_channels: int = 96,
        out_channels: int = 3,
        patch_size: tuple = (2, 4, 4),
    ):
        super().__init__()
        if in_channels < out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must be >= out_channels "
                f"({out_channels})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = tuple(patch_size)

        # ---- Stage 1: patch-size transposed conv (the main upsampler) ----
        # Inverts the encoder's patch embedding. Kernel = stride = patch_size,
        # so output spatial dims are exactly (T, H, W).
        self.upsample = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # ---- Stage 2: repeated 1x1x1 channel reduction ----
        # n = ceil(log2(C / out_channels)). For C=96, out=3 → n=5.
        # Each layer halves the channel count, except the last which lands
        # exactly on out_channels (might be a non-halving step at the end).
        n = math.ceil(math.log2(in_channels / out_channels))
        layers = []
        c = in_channels
        for i in range(n):
            # Halve, but don't drop below out_channels
            next_c = max(c // 2, out_channels)
            layers.append(nn.ConvTranspose3d(c, next_c, kernel_size=1))
            c = next_c
            if c == out_channels:
                break
        # Pan et al.'s spec doesn't mention activations between the 1x1x1 convs.
        # We leave them as pure linear projections — closest to the literal
        # reading of equation (21).
        self.reduce = nn.Sequential(*layers)

        # Initialise transposed convs with truncated normal, biases at zero
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T', H', W', C) channels-last decoder output.

        Returns:
            (B, out_channels, T, H, W) channels-first pixel video.
        """
        # Permute channels-last → channels-first for Conv3d
        # (B, T', H', W', C) → (B, C, T', H', W')
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # Upsample to pixel resolution
        x = self.upsample(x)        # (B, C, T, H, W)
        # Reduce channel count to out_channels
        x = self.reduce(x)          # (B, out_channels, T, H, W)
        return x