"""
Script 02 — Preprocess the full FM dataset.

Reads all .xls files from the configured raw_fm_dir, converts each to
a 256x256x3 RGB spectrogram, and saves the consolidated array as a
single .npy file.

Runtime estimate (on a typical laptop with 8 cores):
    ~5-10 minutes for 10,777 files.

Output: data/processed/fm_spectrograms.npy of shape (N, 256, 256, 3) uint8.

Usage:
    python scripts/02_preprocess_fm.py
    python scripts/02_preprocess_fm.py --workers 4
    python scripts/02_preprocess_fm.py --overwrite
    python scripts/02_preprocess_fm.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time

# Add project root to sys.path so 'src.*' imports work when running directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.preprocessing import preprocess_directory
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing output file')
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = config['paths']['raw_fm_dir']
    output_path = config['paths']['processed_fm']

    print(f"=== FM dataset preprocessing ===")
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
    print(f"Output path:     {summary['output_path']}")
    if summary['failed']:
        print(f"Failures:        {len(summary['failed'])}")


if __name__ == '__main__':
    main()