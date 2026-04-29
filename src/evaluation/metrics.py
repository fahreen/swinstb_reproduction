"""
Per-frame image quality metrics for evaluating spectrogram predictions.

Implements the four metrics Pan et al. report in Section VII (Figure 7):
    - MSE       : raw mean squared error, lower is better
    - PSNR      : peak signal-to-noise ratio in dB, higher is better
    - SSIM      : structural similarity index, higher is better, range [-1, 1]
    - LPIPS     : learned perceptual image patch similarity, lower is better

Notes on input conventions:
    - All metrics operate on RGB images in [0, 1].
    - PSNR and SSIM use data_range=1.0.
    - LPIPS internally expects inputs in [-1, 1]; we map (pred*2 - 1) before
      passing.
    - All metrics are computed per frame, on a (B, 3, H, W) tensor.

LPIPS network choice:
    We use 'alex' (AlexNet-based) per the LPIPS library default. Pan et al.
    don't specify which LPIPS variant they used; 'alex' is the most common
    default in the literature.

Why a class rather than bare functions:
    LPIPS loads a pretrained network (~6 MB AlexNet weights) on first use.
    Reloading per frame would be wasteful. The FrameMetrics class loads it
    once at init and reuses it.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameMetrics:
    """
    Compute per-frame MSE / PSNR / SSIM / LPIPS for predicted vs target images.

    Args:
        device: torch device for the metrics' internal state (LPIPS network).

    Usage:
        metrics = FrameMetrics(device='cuda')
        result = metrics.compute(pred, target)  # returns dict of 4 floats

    Both `pred` and `target` should be (B, 3, H, W) float tensors in [0, 1].
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Lazy-load: import here so the module loads even if torchmetrics/lpips
        # aren't installed; only the FrameMetrics() constructor would fail.
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        import lpips

        # PSNR with data_range=1.0 since images are in [0, 1].
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        # SSIM with data_range=1.0 too.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # LPIPS with AlexNet backbone; freeze and eval mode.
        self.lpips = lpips.LPIPS(net='alex', verbose=False).to(device)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute the four metrics for a batch of frames.

        Args:
            pred:   (B, 3, H, W) predicted RGB frames in [0, 1].
            target: (B, 3, H, W) ground-truth RGB frames in [0, 1].

        Returns:
            Dict with keys 'mse', 'psnr', 'ssim', 'lpips' — each a Python float.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        if pred.dim() != 4 or pred.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W) tensors; got pred.shape={tuple(pred.shape)}"
            )

        # Clamp to valid pixel range. Model output can drift slightly outside
        # [0, 1] since we use MSE without sigmoid; clamping prevents
        # artificial PSNR penalties.
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        # MSE — raw mean over all pixels and channels in the batch
        mse_val = F.mse_loss(pred, target).item()

        # PSNR & SSIM — torchmetrics handles the math; reset state per call
        # so each call is independent (these are cumulative by default).
        self.psnr.reset()
        self.ssim.reset()
        psnr_val = self.psnr(pred, target).item()
        ssim_val = self.ssim(pred, target).item()

        # LPIPS — expects inputs in [-1, 1]. Returns (B, 1, 1, 1) tensor;
        # mean across the batch gives a scalar.
        lpips_in_pred = pred * 2.0 - 1.0
        lpips_in_target = target * 2.0 - 1.0
        lpips_val = self.lpips(lpips_in_pred, lpips_in_target).mean().item()

        return {
            'mse': mse_val,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val,
        }