"""
Script 07 — Generate test-set evaluation results.

Loads a trained model and computes per-frame MSE/PSNR/SSIM/LPIPS curves
over the entire test set.

Outputs:
    - outputs/fm_metrics.npz (or lte_metrics.npz)
    - outputs/figures/framewise_metrics.png (recreates paper Figure 7)
    - outputs/figures/example_predictions.png (sample predictions vs. truth)

Usage:
    python scripts/07_evaluate.py --checkpoint checkpoints/swinstb_fm_best.pt \\
                                   --dataset fm
"""

# TODO: implement
