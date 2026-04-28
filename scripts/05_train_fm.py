"""
Script 05 — Train 3D-SwinSTB on the FM dataset.

Main training entry point. Reads:
    - data/processed/fm_spectrograms.npy
    - configs/default.yaml

Writes:
    - checkpoints/swinstb_fm_latest.pt (every epoch)
    - checkpoints/swinstb_fm_best.pt (when val loss improves)
    - outputs/training_log.csv

Best run on Colab A100. Expected runtime: 10-12 hours for 20 epochs.

Usage:
    python scripts/05_train_fm.py [--config configs/default.yaml] [--resume]
"""

# TODO: implement
