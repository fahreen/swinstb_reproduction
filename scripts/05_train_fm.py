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
    # Fresh run
    python scripts/05_train_fm.py

    # Quick pipeline check (2 epochs × 10 batches)
    python scripts/05_train_fm.py --smoke-test

    # Resume after Colab session timeout
    python scripts/05_train_fm.py --resume

    # Overwrite existing checkpoint and restart
    python scripts/05_train_fm.py --force-restart

    # Use a different config
    python scripts/05_train_fm.py --config configs/my_experiment.yaml
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
    print('3D-SwinSTB training')
    print('=' * 70)
    print(f'Config:        {args.config}')
    print(f'Resume:        {args.resume}')
    print(f'Force restart: {args.force_restart}')
    print(f'Smoke test:    {args.smoke_test}')
    print(f'Seed:          {args.seed}')
    print()

    # Run training
    result = train(
        config=config,
        resume=args.resume,
        force_restart=args.force_restart,
        smoke_test=args.smoke_test,
        seed=args.seed,
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