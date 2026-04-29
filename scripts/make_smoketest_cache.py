"""
Script — Generate a small cache for smoke-testing the training pipeline on CPU.

Reads the first 100 frames from data/processed/fm_spectrograms.npy and
downsamples each from 256x256 to 64x64 via bilinear interpolation,
producing data/processed/fm_smoketest.npy.

The smoke-test cache is used with configs/smoketest.yaml when running:
    python scripts/05_train_fm.py --config configs/smoketest.yaml --smoke-test

This setup runs end-to-end in 1-2 minutes on CPU, validating that all
code paths in the training loop execute correctly. The full 256x256
training is reserved for Colab A100.

Usage:
    python scripts/make_smoketest_cache.py
"""

import os
import sys

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config


def main():
    config = load_config('configs/default.yaml')
    src_path = config['paths']['processed_fm']
    dst_path = src_path.replace('fm_spectrograms.npy', 'fm_smoketest.npy')
    n_frames = 100
    target_size = 64

    print(f"Source:  {src_path}")
    print(f"Output:  {dst_path}")
    print(f"Frames:  {n_frames}")
    print(f"Size:    {target_size}x{target_size}")
    print()

    if not os.path.exists(src_path):
        print(f"ERROR: source cache not found. Run scripts/02_preprocess_fm.py first.")
        sys.exit(1)

    # mmap so we don't load 2 GB just to read 100 frames
    src = np.load(src_path, mmap_mode='r')
    print(f"Source shape: {src.shape}")

    out = np.zeros((n_frames, target_size, target_size, 3), dtype=np.uint8)
    for i in range(n_frames):
        img = Image.fromarray(src[i])
        img_small = img.resize((target_size, target_size), Image.BILINEAR)
        out[i] = np.array(img_small)

    np.save(dst_path, out)
    print(f"Saved: {dst_path}")
    print(f"Shape: {out.shape}, size: {out.nbytes / 1024:.1f} KB")


if __name__ == '__main__':
    main()