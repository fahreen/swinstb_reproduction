# 3D-SwinSTB Reproduction

Reproducing Pan et al. (2025), "Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning," IEEE TWC Vol. 24 No. 1.

**Scope:** 3D-SwinSTB only (spectrogram predictor). 3D-SwinLinear is out of scope.

## Project structure

```
swinstb_reproduction/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── configs/
│   └── default.yaml               # Hyperparameters, paths, training config
│
├── data/                          # Raw and preprocessed data (not checked into git)
│   ├── raw/                       # Original .xls files from NUAA repo
│   │   ├── fm/                    # 10,777 FM-band files
│   │   └── lte/                   # ~2,400 LTE-band files
│   └── processed/                 # Cached preprocessed spectrograms
│       ├── fm_spectrograms.npy    # (10777, 256, 256, 3) uint8 array
│       └── lte_spectrograms.npy   # (~2400, 256, 256, 3) uint8 array
│
├── src/                           # All Python source code
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_iq.py            # Parse raw .xls files
│   │   ├── stft.py                # I/Q → spectrogram conversion
│   │   ├── preprocessing.py       # Batch preprocessing pipeline
│   │   └── dataset.py             # PyTorch Dataset class
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── swinstb.py             # Main model (wraps MONAI SwinUNETR)
│   │   └── projection.py          # 3D Projection Layer (custom output head)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop with AMP + checkpointing
│   │   └── transfer.py            # Transfer learning (FM → LTE)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # MSE, PSNR, SSIM, LPIPS
│   │   └── evaluate.py            # Full test-set evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Load YAML config
│       ├── checkpoint.py          # Save/load checkpoints
│       └── visualization.py       # Plot spectrograms and results
│
├── scripts/                       # Standalone scripts you run directly
│   ├── 01_diagnose_raw_file.py    # Inspect one .xls file
│   ├── 02_preprocess_fm.py        # Batch-process FM dataset → .npy
│   ├── 03_preprocess_lte.py       # Batch-process LTE dataset → .npy
│   ├── 04_verify_model.py         # Forward-pass test with random data
│   ├── 05_train_fm.py             # Train on FM dataset
│   ├── 06_train_lte_transfer.py   # Fine-tune on LTE (transfer learning)
│   └── 07_evaluate.py             # Generate test-set metrics and plots
│
├── notebooks/                     # Jupyter notebooks for exploration (Colab)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_spectrogram_visualization.ipynb
│   ├── 03_model_sanity_check.ipynb
│   └── 04_training_colab.ipynb    # Main training notebook for Colab
│
├── checkpoints/                   # Model weights (not checked into git)
│   ├── swinstb_fm_best.pt         # Best FM-trained weights
│   ├── swinstb_fm_latest.pt       # Latest checkpoint (for resume)
│   └── swinstb_lte_best.pt        # Best LTE fine-tuned weights
│
└── outputs/                       # Evaluation results, plots, logs
    ├── fm_metrics.npz             # Per-frame MSE, PSNR, SSIM, LPIPS on FM test
    ├── lte_metrics.npz            # Same for LTE
    ├── training_log.csv           # Per-epoch loss curves
    └── figures/                   # Plots (frame-wise metrics, example predictions)
```

## Workflow

1. **Setup environment** (local VS Code or Colab): install dependencies from `requirements.txt`.
2. **Place raw data** in `data/raw/fm/` and `data/raw/lte/`.
3. **Diagnose**: run `scripts/01_diagnose_raw_file.py` to verify file format.
4. **Preprocess**: run `scripts/02_preprocess_fm.py` — this creates `data/processed/fm_spectrograms.npy` (takes ~2 hours).
5. **Model sanity check**: run `scripts/04_verify_model.py` — verifies forward pass works.
6. **Train**: run `scripts/05_train_fm.py` (use Colab A100 for this).
7. **Transfer**: run `scripts/06_train_lte_transfer.py` after preprocessing LTE.
8. **Evaluate**: run `scripts/07_evaluate.py` to generate result plots.

## Where to place data

### Local development

```
swinstb_reproduction/
└── data/
    └── raw/
        └── fm/
            ├── 172.19.220.14-2022-0923-092023.xls
            ├── 172.19.220.14-2022-0923-092024.xls
            └── ...
```

### Colab (recommended for training)

Upload the raw data to Google Drive under `My Drive/swinstb_reproduction/data/raw/fm/`. The notebooks will mount Drive automatically and find the files there.

## Reference

See `reference.md` for the complete reproduction reference card with architecture specs, hyperparameters, and open questions.
