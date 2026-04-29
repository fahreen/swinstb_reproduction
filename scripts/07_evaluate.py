"""
Script 07 — Evaluate a trained SwinSTB checkpoint on the test set.

Loads the trained model from a checkpoint, runs it on the held-out test
split, and computes per-frame MSE/PSNR/SSIM/LPIPS metrics. Saves the raw
per-frame arrays to .npz (for plotting) and a human-readable summary
to .json.

Usage:
    # Evaluate best checkpoint on the test set
    python scripts/07_evaluate.py

    # Evaluate a specific checkpoint
    python scripts/07_evaluate.py --checkpoint /path/to/swinstb_fm_best.pt

    # Evaluate on val (sanity) or train (overfitting check)
    python scripts/07_evaluate.py --split val
    python scripts/07_evaluate.py --split train

    # Custom config (e.g. configs/colab.yaml on Colab)
    python scripts/07_evaluate.py --config configs/colab.yaml

Output files (written to <output_dir> from the config):
    eval_<split>_metrics.npz   — four (K,) numpy arrays for plotting
    eval_<split>_summary.json  — per-frame + aggregate mean/std
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
from src.model.swinstb import SwinSTB
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt. Defaults to '
                             '<checkpoint_dir>/swinstb_fm_best.pt')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which dataset split to evaluate on')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint path
    ckpt_path = args.checkpoint or os.path.join(
        config['paths']['checkpoint_dir'], 'swinstb_fm_best.pt'
    )

    print('=' * 70)
    print('SwinSTB Evaluation')
    print('=' * 70)
    print(f'Config:     {args.config}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Split:      {args.split}')
    print(f'Device:     {device}')
    print()

    # ─── Build model ─────────────────────────────────────────────────────────
    model_cfg = config['model']
    model = SwinSTB(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        embed_dim=model_cfg['feature_size'],
        patch_size=tuple(model_cfg['patch_size']),
        window_size=tuple(model_cfg['window_size']),
        encoder_depths=tuple(model_cfg['depths'][:3]),
        encoder_heads=tuple(model_cfg['num_heads'][:3]),
        decoder_depths=(2, 4, 2),
        decoder_heads=(16, 8, 4),
        bottleneck_depth=2,
        mlp_ratio=model_cfg['mlp_ratio'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # ─── Load checkpoint (weights only) ──────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    meta = load_checkpoint(ckpt_path, model, device=str(device))
    print(f"Loaded checkpoint from epoch {meta['epoch']} "
          f"(best val loss: {meta['best_val_loss']:.6f})")
    print()

    # ─── Build dataset + loader ──────────────────────────────────────────────
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

    # ─── Run evaluation ──────────────────────────────────────────────────────
    print()
    print('Running evaluation...')
    metrics = FrameMetrics(device=device)
    results = evaluate_test_set(model, loader, metrics, device, k=K)

    # ─── Save and report ─────────────────────────────────────────────────────
    output_dir = config['paths']['output_dir']
    npz_path = os.path.join(output_dir, f'eval_{args.split}_metrics.npz')
    json_path = os.path.join(output_dir, f'eval_{args.split}_summary.json')
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