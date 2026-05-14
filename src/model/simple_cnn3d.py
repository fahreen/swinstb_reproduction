"""
Simple 3D CNN baseline for spectrogram forecasting.

A deliberately minimal architecture for sanity-checking the data pipeline
and providing a baseline number to compare against the 3D-SwinSTB result.

Architecture:
    Encoder-decoder built from 3D convolutions only. No attention, no skip
    connections, no patch embedding. Spatial downsampling via strided
    Conv3D, upsampling via ConvTranspose3D. Time dimension preserved
    throughout (stride=1 on the time axis).

Parameter count: approximately 1.5M. About 8x smaller than 3D-SwinSTB.

This is NOT a reproduction of any of Pan et al.'s reported baselines.
It is a simple pipeline sanity check: if a basic CNN trains stably and
produces non-trivial outputs on this data, the preprocessing and training
loop are working correctly, isolating SwinSTB-specific issues.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    One 3D conv block: Conv3D + GroupNorm + GELU.

    GroupNorm is used instead of LayerNorm because it works cleanly on 5D
    tensors without requiring explicit shape arithmetic, and instead of
    BatchNorm because batch size is 1 (BatchNorm with batch=1 is degenerate).

    Args:
        in_channels:  input channel count
        out_channels: output channel count
        stride:       spatial stride. Pass (1, 2, 2) to downsample by 2x in
                      spatial dims, keeping time intact. (1, 1, 1) preserves
                      all dims.
    """

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvUpBlock3D(nn.Module):
    """
    Upsampling block: ConvTranspose3D + GroupNorm + GELU.

    ConvTranspose3D (sometimes called "deconv") is the inverse of strided
    Conv3D. With stride (1, 2, 2) and kernel=4, it doubles spatial
    resolution while leaving time intact.
    """

    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(3, 4, 4),
            stride=stride,
            padding=(1, 1, 1),
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.up(x)))


class SimpleCNN3D(nn.Module):
    """
    Simple 3D-CNN encoder-decoder for spectrogram forecasting.

    Input:  (B, 3, 20, 256, 256)
    Output: (B, 3, 20, 256, 256), values in [0, 1]
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, 32, stride=(1, 1, 1))
        self.enc2 = ConvBlock3D(32, 64, stride=(1, 2, 2))
        self.enc3 = ConvBlock3D(64, 64, stride=(1, 1, 1))
        self.enc4 = ConvBlock3D(64, 128, stride=(1, 2, 2))

        self.bottleneck = ConvBlock3D(128, 128, stride=(1, 1, 1))

        self.dec1 = ConvUpBlock3D(128, 64, stride=(1, 2, 2))
        self.dec2 = ConvBlock3D(64, 64, stride=(1, 1, 1))
        self.dec3 = ConvUpBlock3D(64, 32, stride=(1, 2, 2))
        self.dec4 = ConvBlock3D(32, 32, stride=(1, 1, 1))

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.out_conv(x)
        x = self.out_act(x)
        return x