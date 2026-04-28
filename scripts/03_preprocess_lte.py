"""
Script 03 — Preprocess the full LTE dataset.

Same pipeline as 02 but for LTE files (16,254 samples per file).
Output: data/processed/lte_spectrograms.npy.

Runtime estimate: ~2 minutes for ~2,400 files.

Usage:
    python scripts/03_preprocess_lte.py
    python scripts/03_preprocess_lte.py --overwrite
"""

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.preprocessing import preprocess_directory
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = config['paths']['raw_lte_dir']
    output_path = config['paths']['processed_lte']

    print(f"=== LTE dataset preprocessing ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_path}")
    print()

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        print(f"Edit configs/default.yaml to point at your dataset folder.")
        sys.exit(1)

    start = time.time()
    summary = preprocess_directory(
        input_dir=input_dir,
        output_path=output_path,
        num_workers=args.workers,
        overwrite=args.overwrite,
    )
    elapsed = time.time() - start

    print()
    print(f"=== Done in {elapsed:.1f}s ({elapsed/60:.1f} min) ===")
    print(f"Files processed: {summary['n_files']}")
    print(f"Output shape:    {summary['output_shape']}")


if __name__ == '__main__':
    main()