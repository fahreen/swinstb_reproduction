"""
Script 09 — Evaluate a trained Simple 3D-CNN checkpoint on the test set.

Baseline counterpart to scripts/07_evaluate.py. Loads the trained
Simple 3D-CNN from a checkpoint, runs it on the held-out test split, and
computes per-frame MSE/PSNR/SSIM/LPIPS metrics. Saves the raw per-frame
arrays to .npz (for plotting) and a human-readable summary to .json.

The evaluation logic (evaluate_test_set, FrameMetrics, save_metrics,
print_summary) is identical to the SwinSTB evaluation; only the model
class and the default checkpoint path differ. Output files are suffixed
with 'simple_cnn' so they never collide with the SwinSTB eval outputs.

Usage:
    python scripts/09_evaluate_simple_cnn.py
    python scripts/09_evaluate_simple_cnn.py --checkpoint /path/to/simple_cnn_fm_best.pt
    python scripts/09_evaluate_simple_cnn.py --split val
    python scripts/09_evaluate_simple_cnn.py --config configs/colab.yaml
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset import SpectrogramSequenceDataset
from src.evaluation.evaluate import evaluate_test_set, save_metrics, print_summary
from src.evaluation.metrics import FrameMetrics
from src.model.simple_cnn3d import SimpleCNN3D
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default checkpoint is the simple-CNN best checkpoint, not SwinSTB's.
    ckpt_path = args.checkpoint or os.path.join(
        config['paths']['checkpoint_dir'], 'simple_cnn_fm_best.pt'
    )

    print('=' * 70)
    print('Simple 3D-CNN Evaluation')
    print('=' * 70)
    print(f'Config:     {args.config}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Split:      {args.split}')
    print(f'Device:     {device}')
    print()

    # Build the Simple 3D-CNN. Unlike SwinSTB, it has no architecture
    # hyperparameters to read from config beyond in/out channels.
    model_cfg = config['model']
    model = SimpleCNN3D(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    meta = load_checkpoint(ckpt_path, model, device=str(device))
    print(f"Loaded checkpoint from epoch {meta['epoch']} "
          f"(best val loss: {meta['best_val_loss']:.6f})")
    print()

    seq_cfg = config['data']['sequence']
    K = seq_cfg['target_length']
    dataset = SpectrogramSequenceDataset(
        cache_path=config['paths']['processed_fm'],
        split=args.split,
        input_length=seq_cfg['input_length'],
        target_length=K,
    )
    print(f"{args.split.capitalize()} split: {len(dataset)} sequences")

    loader = DataLoader(
        dataset,
        batch_size=config['training'].get('batch_size', 1),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=(device.type == 'cuda'),
    )

    print()
    print('Running evaluation...')
    metrics = FrameMetrics(device=device)
    results = evaluate_test_set(model, loader, metrics, device, k=K)

    # Output files are suffixed with the model name so they sit alongside
    # the SwinSTB eval outputs without overwriting them.
    output_dir = config['paths']['output_dir']
    npz_path = os.path.join(output_dir, f'eval_simple_cnn_{args.split}_metrics.npz')
    json_path = os.path.join(output_dir, f'eval_simple_cnn_{args.split}_summary.json')
    save_metrics(results, npz_path, json_path)

    print()
    print('=' * 70)
    print(f'Evaluation complete on {args.split} split.')
    print('=' * 70)
    print(f'Results saved:')
    print(f'  {npz_path}')
    print(f'  {json_path}')

    print_summary(results)


if __name__ == '__main__':
    main()