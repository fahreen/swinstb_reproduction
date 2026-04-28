"""
Full test-set evaluation.

Computes per-frame metric curves by evaluating on the entire test set
and averaging each metric at each frame position (1 through K=20).

This replicates Pan et al.'s Figure 7 (frame-wise curves).

Functions:
    evaluate_test_set(model, test_loader) -> dict
        Returns dict with keys 'mse', 'psnr', 'ssim', 'lpips', each
        mapping to a length-K numpy array of per-frame averages.
    
    save_metrics(results, output_path) -> None
        Save to .npz for later plotting.
"""

# TODO: implement
