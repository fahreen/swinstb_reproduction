"""
Test-set evaluation: per-frame metrics for SwinSTB predictions.

Pipeline:
    For each test sequence (input 20 frames, target 20 frames):
        1. Run model forward pass → predicted 20 frames.
        2. For each of the K=20 output frames, compute MSE/PSNR/SSIM/LPIPS
           between the predicted frame and the target frame.
        3. Accumulate per-frame values across all test sequences.
    Final result: four arrays of shape (K,) — one mean value per output frame.

This produces the input data for Pan et al.'s Figure 7 — metric vs. predicted
frame index. Early frames (closer to the input) are typically easier to
predict; later frames degrade as the model extrapolates further into the
future.

Memory note:
    We never accumulate all predictions in memory — that would be ~270 GB
    for the full test set. Instead, we compute metrics per batch and
    accumulate scalars only.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import FrameMetrics


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    metrics: FrameMetrics,
    device: torch.device,
    k: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Run the model on the test set and accumulate per-frame metrics.

    Args:
        model: trained SwinSTB. Will be put in eval mode.
        test_loader: DataLoader yielding (input, target) batches, both
            with shape (B, 3, K, H, W).
        metrics: FrameMetrics instance for the metric computations.
        device: torch device for inference.
        k: number of output frames per sequence (default 20).

    Returns:
        Dict with four (k,) numpy arrays — one per metric, each entry
        is the mean over the test set for that frame index.
    """
    model.eval()
    use_amp = device.type == 'cuda'

    # Accumulators: sum of per-frame metric values across batches,
    # plus a count of batches per frame index.
    sums = {
        'mse':   np.zeros(k, dtype=np.float64),
        'psnr':  np.zeros(k, dtype=np.float64),
        'ssim':  np.zeros(k, dtype=np.float64),
        'lpips': np.zeros(k, dtype=np.float64),
    }
    counts = np.zeros(k, dtype=np.int64)
    n_batches = len(test_loader)

    t_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=use_amp)
            targets = targets.to(device, non_blocking=use_amp)

            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu',
                                    enabled=use_amp):
                preds = model(inputs)

            # preds, targets: (B, 3, K, H, W). For each frame index k,
            # extract (B, 3, H, W) slices and compute metrics.
            preds = preds.float()      # cast back to fp32 for metrics
            targets = targets.float()

            for frame_idx in range(k):
                pred_frame = preds[:, :, frame_idx, :, :]
                target_frame = targets[:, :, frame_idx, :, :]
                m = metrics.compute(pred_frame, target_frame)
                # PSNR may be +inf if a batch is identical (unlikely but
                # possible for trivial frames); skip those to avoid
                # poisoning the average.
                if not (np.isnan(m['psnr']) or np.isinf(m['psnr'])):
                    sums['psnr'][frame_idx] += m['psnr']
                    counts[frame_idx] += 1
                else:
                    counts[frame_idx] += 1  # still count for non-PSNR metrics
                sums['mse'][frame_idx]   += m['mse']
                sums['ssim'][frame_idx]  += m['ssim']
                sums['lpips'][frame_idx] += m['lpips']

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
                print(f"  batch {batch_idx+1}/{n_batches}  "
                      f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                      flush=True)

    # Mean across all test batches per frame index.
    results = {
        'mse':   sums['mse']   / counts,
        'psnr':  sums['psnr']  / counts,
        'ssim':  sums['ssim']  / counts,
        'lpips': sums['lpips'] / counts,
    }
    return results


def save_metrics(results: Dict[str, np.ndarray],
                 npz_path: str,
                 json_path: str) -> None:
    """
    Save evaluation results to two complementary files.

    Args:
        results: dict from evaluate_test_set — four (K,) numpy arrays.
        npz_path: where to save the raw per-frame arrays (for plotting later).
        json_path: where to save a human-readable summary (per-frame +
            aggregate mean across all frames).
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **results)

    summary = {
        'per_frame': {
            metric: arr.tolist() for metric, arr in results.items()
        },
        'aggregate_mean': {
            metric: float(arr.mean()) for metric, arr in results.items()
        },
        'aggregate_std': {
            metric: float(arr.std()) for metric, arr in results.items()
        },
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)


def print_summary(results: Dict[str, np.ndarray]) -> None:
    """Pretty-print the per-frame metrics table."""
    k = len(next(iter(results.values())))
    print()
    print(f"{'Frame':<8}{'MSE':>12}{'PSNR (dB)':>12}{'SSIM':>10}{'LPIPS':>10}")
    print('-' * 52)
    for i in range(k):
        print(f"{i:<8}"
              f"{results['mse'][i]:>12.6f}"
              f"{results['psnr'][i]:>12.4f}"
              f"{results['ssim'][i]:>10.4f}"
              f"{results['lpips'][i]:>10.4f}")
    print('-' * 52)
    print(f"{'Mean':<8}"
          f"{results['mse'].mean():>12.6f}"
          f"{results['psnr'].mean():>12.4f}"
          f"{results['ssim'].mean():>10.4f}"
          f"{results['lpips'].mean():>10.4f}")