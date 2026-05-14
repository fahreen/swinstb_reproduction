"""
Script 08 — Train the Simple 3D-CNN baseline on the FM dataset.

Baseline training entry point. Reads:
    - data/processed/fm_spectrograms.npy
    - configs/default.yaml

Writes:
    - checkpoints/simple_cnn_fm_latest.pt (every epoch)
    - checkpoints/simple_cnn_fm_best.pt (when val loss improves)
    - outputs/training_log_simple_cnn.csv

This trains the lightweight 3D-CNN baseline (src/model/simple_cnn3d.py)
rather than the full 3D-SwinSTB. It exists as a pipeline sanity check and
to produce a comparison row against the SwinSTB reproduction. The model
is about 8x smaller than SwinSTB, so each epoch is faster.

The actual training logic is shared with the SwinSTB run: this script
just calls the same train() function from src/training/trainer.py with
model_type='simple_cnn', which swaps the model class and the output
filenames while keeping the optimizer, loss, AMP, early stopping, and
checkpoint logic identical.

Usage:
    # Fresh run
    python scripts/08_train_simple_cnn.py

    # Quick pipeline check (2 epochs x 10 batches)
    python scripts/08_train_simple_cnn.py --smoke-test

    # Resume after Colab session timeout
    python scripts/08_train_simple_cnn.py --resume

    # Overwrite existing checkpoint and restart
    python scripts/08_train_simple_cnn.py --force-restart

    # Use a different config
    python scripts/08_train_simple_cnn.py --config configs/my_experiment.yaml
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.training.trainer import train
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing _latest.pt checkpoint')
    parser.add_argument('--force-restart', action='store_true',
                        help='Overwrite existing checkpoint and start fresh')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick pipeline verification: 2 epochs x 10 batches')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for torch/numpy reproducibility')
    args = parser.parse_args()

    if args.resume and args.force_restart:
        print('ERROR: --resume and --force-restart are mutually exclusive.')
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    print('=' * 70)
    print('Simple 3D-CNN baseline training')
    print('=' * 70)
    print(f'Config:        {args.config}')
    print(f'Resume:        {args.resume}')
    print(f'Force restart: {args.force_restart}')
    print(f'Smoke test:    {args.smoke_test}')
    print(f'Seed:          {args.seed}')
    print()

    # Run training. The only difference from scripts/05_train_fm.py is
    # model_type='simple_cnn', which tells the shared trainer to build
    # SimpleCNN3D instead of SwinSTB and to use simple_cnn_fm_* filenames.
    result = train(
        config=config,
        resume=args.resume,
        force_restart=args.force_restart,
        smoke_test=args.smoke_test,
        seed=args.seed,
        model_type='simple_cnn',
    )

    # Final summary
    print()
    print('=' * 70)
    print('Training complete.')
    print('=' * 70)
    print(f'Final epoch:    {result["final_epoch"]}')
    print(f'Best val loss:  {result["best_val_loss"]:.6f}')
    print(f'Best epoch:     {result["best_epoch"]}')
    print(f'Total epochs:   {len(result["history"])}')

    if result['history']:
        first = result['history'][0]
        last = result['history'][-1]
        print(f'First epoch val loss: {first["val_loss"]:.6f}')
        print(f'Last epoch val loss:  {last["val_loss"]:.6f}')


if __name__ == '__main__':
    main()