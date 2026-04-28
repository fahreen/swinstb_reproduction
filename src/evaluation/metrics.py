"""
Per-frame evaluation metrics.

Per Pan et al. Section VI-A equations (30)-(33):
    MSE   — sum of squared pixel differences
    PSNR  — peak signal-to-noise ratio (higher is better)
    SSIM  — structural similarity (higher is better, max 1.0)
    LPIPS — learned perceptual similarity (lower is better)

Implemented using:
    - F.mse_loss (for MSE)
    - torchmetrics.PeakSignalNoiseRatio
    - torchmetrics.StructuralSimilarityIndexMeasure
    - lpips package (LearnedPerceptualImagePatchSimilarity)

Functions:
    compute_frame_metrics(pred, target) -> dict
        Compute all 4 metrics for one (B, C, H, W) frame pair.
"""

# TODO: implement
