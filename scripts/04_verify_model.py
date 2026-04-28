"""
Script 04 — Verify full SwinSTB forward pass.

Runs a random input through the full model (encoder + decoder + projection)
and verifies:
    - The output shape exactly matches the input shape.
    - The output is channels-first PyTorch convention.
    - Total parameter count is reasonable (~10-20M for paper config).

Usage:
    python scripts/04_verify_model.py
    python scripts/04_verify_model.py --batch 1 --time 20 --size 256
    python scripts/04_verify_model.py --device cuda
    python scripts/04_verify_model.py --time 8 --size 64    # tiny test
"""

import argparse
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.model.swinstb import SwinSTB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--time', type=int, default=20)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = torch.device(args.device)

    print('=== SwinSTB full model verification ===')
    print(f'Device:       {device}')
    print(f'Input shape:  ({args.batch}, 3, {args.time}, {args.size}, {args.size})')
    print()

    # Pan et al. defaults
    model = SwinSTB(
        in_channels=3,
        out_channels=3,
        embed_dim=96,
        patch_size=(2, 4, 4),
        window_size=(2, 7, 7),
        encoder_depths=(2, 4, 2),
        encoder_heads=(4, 8, 16),
        decoder_depths=(2, 4, 2),
        decoder_heads=(16, 8, 4),
        bottleneck_depth=2,
        mlp_ratio=2.0,
    ).to(device)
    model.eval()

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters())
    n_projection = sum(p.numel() for p in model.projection.parameters())
    print(f'Total parameters:      {n_params:>12,}')
    print(f'  Encoder:             {n_encoder:>12,}')
    print(f'  Decoder:             {n_decoder:>12,}')
    print(f'  Projection:          {n_projection:>12,}')
    print(f'Reference (Pan et al. paper): ~16.32M total')
    print()

    # Random input
    x = torch.randn(args.batch, 3, args.time, args.size, args.size, device=device)
    expected_output_shape = tuple(x.shape)
    print(f'Running forward pass...')
    with torch.no_grad():
        y = model(x)

    print()
    print('=== Output shape check ===')
    got = tuple(y.shape)
    match = got == expected_output_shape
    marker = 'OK ' if match else 'BAD'
    print(f'  [{marker}] output: {got}  expected: {expected_output_shape}')
    print()

    # Value range sanity (just print, don't enforce — output isn't bounded)
    print(f'Output value range: [{y.min().item():.3f}, {y.max().item():.3f}]')
    print(f'Output mean:        {y.mean().item():.3f}')
    print(f'Output std:         {y.std().item():.3f}')
    print('(Note: untrained model produces noise; range is not bounded yet)')
    print()

    if match:
        print('SUCCESS — full model forward pass verified.')
    else:
        print('SHAPE MISMATCH — investigate.')
        sys.exit(1)


if __name__ == '__main__':
    main()