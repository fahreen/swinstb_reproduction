"""
Script 01 — Diagnose a single raw .xls file from the NUAA dataset.

Verifies file format, checks I/Q value distributions, runs an STFT,
identifies the most active frequency bins, and saves a 4-panel diagnostic
figure plus the final 256x256x3 RGB spectrogram as PNG.

Usage:
    python scripts/01_diagnose_raw_file.py <path-to-xls-file> [--output-dir DIR]

Example:
    python scripts/01_diagnose_raw_file.py "C:/Users/fahre/Documents/spectrascope/data/FM-Spectrum-Dataset/172.19.220.14-2022-0923-092023.xls"
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Allow running this script from any directory by adding the project root
# to sys.path. This is a small convenience for direct invocation.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.parse_iq import (
    parse_iq_file,
    get_file_timestamp,
    FM_SAMPLES_PER_FILE,
    LTE_SAMPLES_PER_FILE,
    ADC_MIN,
    ADC_MAX,
)
from src.data.stft import iq_to_spectrogram, compute_stft_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filepath', type=str, help='Path to a NUAA .xls file')
    parser.add_argument('--output-dir', type=str, default='./outputs/figures',
                        help='Where to save diagnostic figures')
    parser.add_argument('--center-freq-mhz', type=float, default=99.0,
                        help='RF centre frequency in MHz (99 for FM, 700 for LTE)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — Parse and validate
    # ─────────────────────────────────────────────────────────────────────────
    print(f"=== Diagnosing {args.filepath} ===")
    print()

    timestamp = get_file_timestamp(args.filepath)
    print(f"Filename timestamp: {timestamp}")

    iq = parse_iq_file(args.filepath)
    print(f"Total samples: {len(iq)}")
    if len(iq) == FM_SAMPLES_PER_FILE:
        print(f"  → matches FM_SAMPLES_PER_FILE ({FM_SAMPLES_PER_FILE})")
    elif len(iq) == LTE_SAMPLES_PER_FILE:
        print(f"  → matches LTE_SAMPLES_PER_FILE ({LTE_SAMPLES_PER_FILE})")
    else:
        print(f"  → unexpected sample count")

    i_samples = iq.real
    q_samples = iq.imag
    print()
    print(f"=== I/Q value distributions ===")
    print(f"I: min={i_samples.min():.0f}, max={i_samples.max():.0f}, "
          f"mean={i_samples.mean():.1f}, std={i_samples.std():.1f}")
    print(f"Q: min={q_samples.min():.0f}, max={q_samples.max():.0f}, "
          f"mean={q_samples.mean():.1f}, std={q_samples.std():.1f}")

    in_range = ((i_samples >= ADC_MIN) & (i_samples <= ADC_MAX)).all() and \
               ((q_samples >= ADC_MIN) & (q_samples <= ADC_MAX)).all()
    print(f"All values in 14-bit signed range [{ADC_MIN}, {ADC_MAX}]: {in_range}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — STFT diagnostics
    # ─────────────────────────────────────────────────────────────────────────
    print(f"=== STFT diagnostics ===")
    f_mhz, t, power = compute_stft_raw(
        iq, center_freq_hz=args.center_freq_mhz * 1e6
    )
    power_db = 10 * np.log10(power + 1e-12)

    print(f"STFT output shape: {power.shape} (freq bins × time windows)")
    print(f"Frequency range: {f_mhz[0]:.3f} MHz to {f_mhz[-1]:.3f} MHz")
    print(f"Resolution: {(f_mhz[1] - f_mhz[0]) * 1000:.2f} kHz/bin")
    print(f"Time span: {t[0]*1000:.3f} ms to {t[-1]*1000:.3f} ms "
          f"(total {(t[-1]-t[0])*1000:.3f} ms = "
          f"{len(iq)/31_250_000*1000:.3f} ms expected)")
    print()

    # Top active bins
    mean_power = power.mean(axis=1)
    top10 = np.argsort(mean_power)[-10:][::-1]
    print(f"=== Top 10 most active frequency bins ===")
    for bin_idx in top10:
        freq = f_mhz[bin_idx]
        db = 10 * np.log10(mean_power[bin_idx] + 1e-12)
        print(f"  Bin {bin_idx:3d}: {freq:7.3f} MHz, mean power {db:7.2f} dB")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — Pipeline output (the actual model input)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"=== Final pipeline output ===")
    rgb = iq_to_spectrogram(iq)
    print(f"Shape: {rgb.shape}, dtype: {rgb.dtype}, value range: "
          f"[{rgb.min()}, {rgb.max()}]")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — Save visualisations
    # ─────────────────────────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.filepath))[0]

    # 4-panel diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    im = ax.imshow(power, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t[0]*1000, t[-1]*1000, f_mhz[0], f_mhz[-1]])
    ax.set_title(f'Raw power ({power.shape[0]} bins × {power.shape[1]} windows)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    plt.colorbar(im, ax=ax, label='Power (linear)')

    ax = axes[0, 1]
    im = ax.imshow(power_db, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t[0]*1000, t[-1]*1000, f_mhz[0], f_mhz[-1]])
    ax.set_title('Log-compressed (dB)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    plt.colorbar(im, ax=ax, label='Power (dB)')

    ax = axes[1, 0]
    ax.plot(f_mhz, 10 * np.log10(mean_power + 1e-12))
    if abs(args.center_freq_mhz - 99.0) < 1:
        ax.axvspan(88, 108, alpha=0.2, color='green',
                   label='FM broadcast band')
    ax.axvline(args.center_freq_mhz, color='red', linestyle='--',
               label=f'Centre freq ({args.center_freq_mhz} MHz)')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Mean power (dB)')
    ax.set_title('Average power per frequency bin')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.imshow(rgb, aspect='auto', origin='upper')
    ax.set_title(f'Final pipeline output (256×256 RGB Jet)')
    ax.set_xlabel('Pixel x')
    ax.set_ylabel('Pixel y')

    plt.tight_layout()
    diag_path = os.path.join(args.output_dir, f'{base}_diagnostic.png')
    plt.savefig(diag_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {diag_path}")

    # Save the final RGB spectrogram alone, exactly as it would be fed to the model
    rgb_path = os.path.join(args.output_dir, f'{base}_input_frame.png')
    Image.fromarray(rgb).save(rgb_path)
    print(f"Saved: {rgb_path}")
    print()
    print("Done.")


if __name__ == '__main__':
    main()