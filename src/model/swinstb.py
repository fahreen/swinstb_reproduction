"""
3D-SwinSTB — top-level model, wiring encoder + decoder + projection.

This is the full reproduction of Pan et al. (2025), "Spectrum Prediction
With Deep 3D Pyramid Vision Transformer Learning," IEEE TWC.

Forward pass:
    Input  (B, 3, T, H, W) channels-first RGB spectrogram video
    →  encoder produces s1_en, s2_en, s3_en, x_de  (channels-last)
    →  decoder concatenates skips and Patch-Expands back to patch-grid res
    →  projection layer upsamples to pixel res and reduces channels to 3
    Output (B, 3, T, H, W) channels-first predicted future RGB video

For the paper's defaults (T=20, H=W=256, C=96), every stage's tensor shape is
locked by Pan et al. Section IV-A. We verify in scripts/04_verify_model.py
that the forward pass produces an output identical in shape to the input.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .video_swin_encoder import VideoSwinEncoder
from .decoder import Decoder3D
from .projection import ProjectionLayer3D


class SwinSTB(nn.Module):
    """
    Full 3D-SwinSTB model.

    Args:
        in_channels: input channels (3 for RGB).
        out_channels: output channels (3 for RGB).
        embed_dim: base feature dim C. Default 96.
        patch_size: 3D patch (Tp, Hp, Wp). Default (2, 4, 4).
        window_size: 3D window (P, M, M). Default (2, 7, 7).
        encoder_depths: blocks per encoder stage. Default (2, 4, 2).
        encoder_heads: heads per encoder stage. Default (4, 8, 16).
        decoder_depths: blocks per decoder stage. Default (2, 4, 2).
        decoder_heads: heads per decoder stage. Default (16, 8, 4).
        bottleneck_depth: blocks in the bottleneck. Default 2.
        mlp_ratio: MLP expansion ratio. Default 2.0 (Pan et al. eq. 14).
        stochastic_depth_prob: drop-path probability. Default 0.0.

    Forward:
        Input:  (B, in_channels, T, H, W)
        Output: (B, out_channels, T, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        window_size: Tuple[int, int, int] = (2, 7, 7),
        encoder_depths: Tuple[int, int, int] = (2, 4, 2),
        encoder_heads: Tuple[int, int, int] = (4, 8, 16),
        decoder_depths: Tuple[int, int, int] = (2, 4, 2),
        decoder_heads: Tuple[int, int, int] = (16, 8, 4),
        bottleneck_depth: int = 2,
        mlp_ratio: float = 2.0,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()

        self.encoder = VideoSwinEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            window_size=window_size,
            depths=encoder_depths,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            bottleneck_depth=bottleneck_depth,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        self.decoder = Decoder3D(
            embed_dim=embed_dim,
            depths=decoder_depths,
            num_heads=decoder_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )

        self.projection = ProjectionLayer3D(
            in_channels=embed_dim,
            out_channels=out_channels,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T, H, W) input video (channels-first).

        Returns:
            (B, out_channels, T, H, W) predicted output (channels-first).
        """
        # Encoder produces channels-last skip features and a bottleneck
        s1_en, s2_en, s3_en, x_de = self.encoder(x)

        # Decoder concatenates skips and patch-expands back to (B, T', 4H', 4W', C)
        decoded = self.decoder(x_de, s3_en, s2_en, s1_en)

        # Projection: channels-last → channels-first, upsample to pixel res,
        # reduce channels to out_channels
        out = self.projection(decoded)
        return out


def create_model(config: dict = None) -> SwinSTB:
    """
    Build a SwinSTB instance from a config dict (or with paper defaults).

    Args:
        config: optional dict with keys matching SwinSTB constructor.

    Returns:
        Initialised SwinSTB model on CPU.
    """
    if config is None:
        return SwinSTB()

    # Allow nested config['model'] or flat config
    model_cfg = config.get('model', config)

    return SwinSTB(
        in_channels=model_cfg.get('in_channels', 3),
        out_channels=model_cfg.get('out_channels', 3),
        embed_dim=model_cfg.get('feature_size', 96),
        patch_size=tuple(model_cfg.get('patch_size', (2, 4, 4))),
        window_size=tuple(model_cfg.get('window_size', (2, 7, 7))),
        encoder_depths=tuple(model_cfg.get('depths', (2, 4, 2, 2))[:3]),
        encoder_heads=tuple(model_cfg.get('num_heads', (4, 8, 16, 16))[:3]),
        decoder_depths=tuple(model_cfg.get('decoder_depths', (2, 4, 2))),
        decoder_heads=tuple(model_cfg.get('decoder_heads', (16, 8, 4))),
        bottleneck_depth=model_cfg.get('bottleneck_depth', 2),
        mlp_ratio=model_cfg.get('mlp_ratio', 2.0),
        stochastic_depth_prob=model_cfg.get('stochastic_depth_prob', 0.0),
    )