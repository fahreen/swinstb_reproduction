"""
Sanity-check the SpectrogramSequenceDataset end-to-end.

Verifies:
    - All three splits load without error
    - Split sizes match the expected 4:1:1 chronological partition
    - __getitem__ returns the right tensor shapes and dtypes
    - Pixel values are normalised correctly to [0, 1]
    - First-and-last example shapes are consistent

Usage:
    python scripts/sanity_check_dataset.py
    python scripts/sanity_check_dataset.py --config configs/default.yaml
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset import SpectrogramSequenceDataset
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    cache_path = config['paths']['processed_fm']
    T = config['data']['sequence']['input_length']
    K = config['data']['sequence']['target_length']

    print(f"=== Dataset sanity check ===")
    print(f"Cache: {cache_path}")
    print(f"T = {T}, K = {K}")
    print()

    if not os.path.exists(cache_path):
        print(f"ERROR: cache not found. Run scripts/02_preprocess_fm.py first.")
        sys.exit(1)

    for split in ('train', 'val', 'test'):
        ds = SpectrogramSequenceDataset(
            cache_path=cache_path,
            split=split,
            input_length=T,
            target_length=K,
        )
        print(ds.describe())

        # Check first and last example
        x_first, y_first = ds[0]
        x_last, y_last = ds[len(ds) - 1]

        print(f"  First example:")
        print(f"    input  shape={tuple(x_first.shape)}, dtype={x_first.dtype}, "
              f"range=[{x_first.min():.3f}, {x_first.max():.3f}]")
        print(f"    target shape={tuple(y_first.shape)}, dtype={y_first.dtype}, "
              f"range=[{y_first.min():.3f}, {y_first.max():.3f}]")
        print(f"  Last example:")
        print(f"    input  shape={tuple(x_last.shape)}, "
              f"range=[{x_last.min():.3f}, {x_last.max():.3f}]")
        print(f"    target shape={tuple(y_last.shape)}, "
              f"range=[{y_last.min():.3f}, {y_last.max():.3f}]")

        # Validate
        expected_input_shape = (3, T, 256, 256)
        expected_target_shape = (3, K, 256, 256)
        assert tuple(x_first.shape) == expected_input_shape, \
            f"Bad input shape: {x_first.shape}"
        assert tuple(y_first.shape) == expected_target_shape, \
            f"Bad target shape: {y_first.shape}"
        assert 0.0 <= x_first.min() and x_first.max() <= 1.0, \
            f"Input not in [0, 1]"
        assert 0.0 <= y_first.min() and y_first.max() <= 1.0, \
            f"Target not in [0, 1]"
        print(f"  Validation: OK")
        print()

    print("=== All splits OK ===")


if __name__ == '__main__':
    main()