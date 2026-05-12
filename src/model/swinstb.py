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
# torch.nn is the neural network building blocks module.
# Contains: layers (Linear, Conv3d, LayerNorm, ...), losses (MSELoss, ...),
# the base Module class, and lots of activation functions.
# By convention always imported as "nn" — saves typing.

from .video_swin_encoder import VideoSwinEncoder
from .decoder import Decoder3D
from .projection import ProjectionLayer3D
# Relative imports — these refer to the three sibling files in this same
# src/model/ folder. We'll walk through each next.


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
        # embed_dim (C) = the base feature dimension.
        # After patch embedding, every "token" in the model is represented
        # by a vector of this many numbers. Stage 1 uses C=96 channels.
        # Stage 2 doubles to 192, stage 3 to 384.
        # Larger C = more model capacity but more compute and parameters.

        patch_size: Tuple[int, int, int] = (2, 4, 4),
        # Patch size (Tp, Hp, Wp).
        # Tp=2: every two consecutive time frames become one patch.
        # Hp=Wp=4: every 4x4 pixel block becomes one patch.
        # So the (T=20, H=256, W=256) input → patch grid (10, 64, 64).

        window_size: Tuple[int, int, int] = (2, 7, 7),
        # Swin attention window (P, M, M).
        # Attention runs WITHIN each window only.
        # P=2: window covers 2 time-patches (so 4 raw frames worth).
        # M=7: each window covers 7x7 = 49 spatial patches.
        # 7 is the standard Swin window size from the original paper.

        encoder_depths: Tuple[int, int, int] = (2, 4, 2),
        # Number of Swin Transformer blocks PER ENCODER STAGE.
        # Stage 1: 2 blocks. Stage 2: 4 blocks. Stage 3: 2 blocks.
        # More blocks = more capacity for that resolution.
        # Pan et al. used (2, 4, 2) — the middle stage gets the most blocks.

        encoder_heads: Tuple[int, int, int] = (4, 8, 16),
        # Number of attention HEADS per stage.
        # An attention head is one parallel attention computation.
        # Multi-head attention lets the model attend to different patterns
        # in parallel — e.g., one head focused on local edges, another on
        # global structure.
        # Stage 1: 4 heads. Stage 2: 8 heads. Stage 3: 16 heads.
        # Heads typically scale with channel count (more channels = more heads).

        decoder_depths: Tuple[int, int, int] = (2, 4, 2),
        decoder_heads: Tuple[int, int, int] = (16, 8, 4),
        # Decoder mirrors the encoder. Same depths, REVERSED head counts.
        # At its top (the part closest to the bottleneck), the decoder
        # operates on the highest channel count (16 heads matches 16 heads
        # in encoder stage 3). At its bottom (closest to output), it
        # operates on the lowest (4 heads matches encoder stage 1).

        bottleneck_depth: int = 2,
        # Bottleneck = the very bottom of the U-shape. After all encoder
        # stages but before the decoder starts. Runs at the smallest
        # spatial resolution. 2 more Swin blocks here.

        mlp_ratio: float = 2.0,
        # Each Swin block has two parts: attention + MLP.
        # The MLP temporarily expands the channel count by mlp_ratio,
        # applies a nonlinearity, then projects back.
        # ratio=2.0 → temporary expansion to 2C channels.
        # Standard Swin used 4.0; Pan et al. eq. 14 specifies 2.0 here.

        stochastic_depth_prob: float = 0.0,
        # "Stochastic depth" = randomly skip some blocks during training
        # to regularize the network. 0.0 means never skip — no regularization.
        # Pan et al. don't specify this; we set it off.
    ):
        super().__init__()
        # CRUCIAL: every nn.Module subclass MUST call super().__init__().
        # This initializes the parent class's internal bookkeeping —
        # parameter registration, hook lists, etc. Without it, PyTorch
        # can't track this module's parameters or move them to GPU.

        # ─── Build the encoder ────────────────────────────────────────────
        # Three encoder stages + bottleneck. Output:
        #   x_de  : the bottleneck features (deepest, smallest spatial)
        #   s3_en : skip from end of stage 3 (last skip, smallest)
        #   s2_en : skip from end of stage 2 (medium)
        #   s1_en : skip from end of stage 1 (largest, finest detail)
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

        # ─── Build the decoder ────────────────────────────────────────────
        # Three decoder stages. Each:
        #   1. Concatenates a skip from the encoder.
        #   2. Projects back to natural channel count.
        #   3. Runs Swin blocks.
        #   4. Patch-expands (doubles spatial size, halves channels).
        self.decoder = Decoder3D(
            embed_dim=embed_dim,
            depths=decoder_depths,
            num_heads=decoder_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )

        # ─── Build the projection ─────────────────────────────────────────
        # Final layer that converts the decoder's high-dim feature tensor
        # back to a 3-channel RGB image at the original (T, H, W) resolution.
        self.projection = ProjectionLayer3D(
            in_channels=embed_dim,
            out_channels=out_channels,
            patch_size=patch_size,
        )

        # NOTE: Assigning these three submodules to self.encoder/decoder/projection
        # automatically registers them with PyTorch. Their parameters now appear
        # in self.parameters(), they move with self.to('cuda'), etc.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T, H, W) input video (channels-first).

        Returns:
            (B, out_channels, T, H, W) predicted output (channels-first).
        """
        # The entire forward pass is just three calls:
        # encoder → decoder → projection. Each submodule handles its
        # own internal complexity. This is the strength of modular design.

        # ─── Encoder: input → bottleneck + skip features ─────────────────
        # The encoder returns a TUPLE of 4 tensors:
        #   s1_en: skip features from after encoder stage 1 (highest res)
        #   s2_en: skip features from after encoder stage 2
        #   s3_en: skip features from after encoder stage 3 (lowest res)
        #   x_de : bottleneck features (deepest, will go to decoder)
        # All are in channels-last format (B, T', H', W', C') for the
        # internal Swin operations.
        s1_en, s2_en, s3_en, x_de = self.encoder(x)

        # ─── Decoder: bottleneck + skips → upsampled features ────────────
        # Decoder takes the bottleneck features and progressively expands.
        # At each stage, it concatenates with the corresponding skip.
        # Note the REVERSE order: s3_en is consumed first (smallest, matches
        # decoder's first stage), then s2_en, then s1_en.
        decoded = self.decoder(x_de, s3_en, s2_en, s1_en)
        # decoded shape (channels-last): (B, T, 4*H', 4*W', C) where
        # H'=H/patch_h=64, so 4*H'=256 — back to pixel-grid resolution.
        # Wait: actually still at patch grid (10, 64, 64) here per docstring;
        # the projection does the final upsample.

        # ─── Projection: features → final RGB output ─────────────────────
        # The projection layer does two things:
        # 1. Upsamples back to full (T, H, W) pixel resolution via
        #    transposed convolution.
        # 2. Reduces channels from embed_dim (96) down to out_channels (3).
        # Returns channels-first format, matching the original input shape.
        out = self.projection(decoded)
        return out
        # out shape: (B, 3, 20, 256, 256) — same as input shape.
        # This is the predicted future spectrogram movie.


def create_model(config: dict = None) -> SwinSTB:
    """
    Build a SwinSTB instance from a config dict (or with paper defaults).

    Args:
        config: optional dict with keys matching SwinSTB constructor.

    Returns:
        Initialised SwinSTB model on CPU.
    """
    # Convenience factory function.
    # Lets you build a model from a YAML config (e.g., configs/default.yaml).
    # Used by the training script so you can override defaults without
    # editing code.

    if config is None:
        return SwinSTB()
        # No config → use all defaults from the class.

    # Allow nested config['model'] or flat config.
    # Some configs look like {model: {in_channels: 3, ...}}, others are flat.
    model_cfg = config.get('model', config)

    return SwinSTB(
        in_channels=model_cfg.get('in_channels', 3),
        out_channels=model_cfg.get('out_channels', 3),
        embed_dim=model_cfg.get('feature_size', 96),
        # .get(key, default) returns config[key] if it exists, else default.
        # This makes the config tolerant of missing keys.

        patch_size=tuple(model_cfg.get('patch_size', (2, 4, 4))),
        window_size=tuple(model_cfg.get('window_size', (2, 7, 7))),
        # tuple(...) ensures the config value (which YAML might load as list)
        # is converted to a tuple — the class expects tuples for immutability.

        encoder_depths=tuple(model_cfg.get('depths', (2, 4, 2, 2))[:3]),
        encoder_heads=tuple(model_cfg.get('num_heads', (4, 8, 16, 16))[:3]),
        # Note the [:3] slicing: config might have 4 entries (3 encoder
        # stages + 1 bottleneck), but the encoder constructor takes only
        # 3 for the stages.

        decoder_depths=tuple(model_cfg.get('decoder_depths', (2, 4, 2))),
        decoder_heads=tuple(model_cfg.get('decoder_heads', (16, 8, 4))),
        bottleneck_depth=model_cfg.get('bottleneck_depth', 2),
        mlp_ratio=model_cfg.get('mlp_ratio', 2.0),
        stochastic_depth_prob=model_cfg.get('stochastic_depth_prob', 0.0),
    )