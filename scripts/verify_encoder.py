"""
Verify the VideoSwinEncoder forward pass produces the expected feature maps.

Runs a small (random) input through the encoder and checks:
    - All four outputs have the right shape.
    - Encoder output S3_en and bottleneck X_de match in shape.
    - Stride math from Pan et al. holds end-to-end.

This is intentionally minimal — we run on CPU with a small input and
just check shapes, not training behaviour. Once the encoder works,
we move on to the decoder.

Usage:
    python scripts/verify_encoder.py
    python scripts/verify_encoder.py --batch 1 --time 20 --size 256
    python scripts/verify_encoder.py --device cuda    # if you have a GPU
"""

import argparse
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.model.video_swin_encoder import VideoSwinEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1, help='Batch size for the test input')
    parser.add_argument('--time', type=int, default=20, help='Number of input frames T')
    parser.add_argument('--size', type=int, default=256, help='Spatial resolution H = W')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = torch.device(args.device)

    print('=== VideoSwinEncoder verification ===')
    print(f'Device:       {device}')
    print(f'Input shape:  ({args.batch}, 3, {args.time}, {args.size}, {args.size})')
    print()

    # Pan et al. defaults
    encoder = VideoSwinEncoder(
        in_channels=3,
        embed_dim=96,
        patch_size=(2, 4, 4),
        window_size=(2, 7, 7),
        depths=(2, 4, 2),
        num_heads=(4, 8, 16),
        mlp_ratio=2.0,
        bottleneck_depth=2,
    ).to(device)
    encoder.eval()

    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'Total parameters:     {n_params:,}')
    print(f'Trainable parameters: {n_trainable:,}')
    print()

    # Random input
    x = torch.randn(args.batch, 3, args.time, args.size, args.size, device=device)
    print(f'Running forward pass...')
    with torch.no_grad():
        s1, s2, s3, x_de = encoder(x)

    # Expected shapes from Pan et al.'s arithmetic:
    #   T' = T / Tp = time / 2
    #   H' = H / Hp = size / 4
    #   W' = W / Wp = size / 4
    Tp, Hp, Wp = encoder.patch_size
    T_, H_, W_ = args.time // Tp, args.size // Hp, args.size // Wp
    expected_s1 = (args.batch, T_, H_,     W_,     96)
    expected_s2 = (args.batch, T_, H_ // 2, W_ // 2, 192)
    expected_s3 = (args.batch, T_, H_ // 4, W_ // 4, 384)
    expected_de = expected_s3  # bottleneck preserves shape

    # Print and verify
    def fmt(name, got, expected):
        match = tuple(got.shape) == expected
        marker = 'OK ' if match else 'BAD'
        print(f'  [{marker}] {name}: shape={tuple(got.shape)}  expected={expected}')
        return match

    print()
    print('=== Output shapes ===')
    ok = True
    ok &= fmt('S1_en (encoder stage 1)', s1, expected_s1)
    ok &= fmt('S2_en (encoder stage 2)', s2, expected_s2)
    ok &= fmt('S3_en (encoder stage 3)', s3, expected_s3)
    ok &= fmt('X_de  (bottleneck)     ', x_de, expected_de)

    print()
    if ok:
        print('All shapes match Pan et al. specification.')
    else:
        print('SHAPE MISMATCH — investigate before proceeding.')
        sys.exit(1)


if __name__ == '__main__':
    main()