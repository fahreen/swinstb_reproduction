"""
Script 06 — Transfer learning: fine-tune FM-trained model on LTE.

Reads:
    - checkpoints/swinstb_fm_best.pt (source weights)
    - data/processed/lte_spectrograms.npy

Writes:
    - checkpoints/swinstb_lte_best.pt
    - outputs/transfer_log.csv

Per Pan et al., should converge in ~77% less time than from-scratch.

Usage:
    python scripts/06_train_lte_transfer.py [--config configs/default.yaml]
"""

# TODO: implement
