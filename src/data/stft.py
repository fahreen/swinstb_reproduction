"""
Convert complex I/Q samples into 256x256x3 RGB spectrograms.

Pipeline (reproducing Pan et al. Section VI-A as closely as possible):

    1. STFT with Hann window, length 256, no overlap.
       Effective sample rate = 31.25 MHz (125 MHz / decimation 4).
       Two-sided STFT (return_onesided=False) since the signal is complex
       baseband — both sides of the spectrum are independent and meaningful.

       For an FM file (32508 samples):  output shape is (256, 126).
       For an LTE file (16254 samples): output shape is (256, 63).

    2. fftshift along the frequency axis to put negative frequencies first,
       producing a monotonically-increasing frequency axis from
       (centre - fs/2) to (centre + fs/2 - df).

    3. Magnitude-squared to get power.

    4. Log compression (10*log10(power + epsilon)).
       Paper is silent on this; standard practice for spectrograms because
       the linear power spans many orders of magnitude.

    5. Per-file min-max normalisation to [0, 1].

    6. Resize to 256x256 via bilinear interpolation.

    7. Apply Jet colormap → RGB uint8 (Pan et al. Appendix A confirms Jet).

    Returns: (256, 256, 3) uint8 array.
"""

from typing import Tuple

import matplotlib
import numpy as np
import scipy.signal
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Default STFT configuration (Pan et al. Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLING_FREQ = 125_000_000  # Hz, before decimation
DEFAULT_DECIMATION = 4
DEFAULT_WINDOW_LENGTH = 256
DEFAULT_WINDOW_TYPE = 'hann'
DEFAULT_NOVERLAP = 0  # No overlap → 126 time windows for 32508 samples

DEFAULT_OUTPUT_SIZE = 256  # H = W = 256 per paper
DEFAULT_COLORMAP = 'jet'
LOG_EPSILON = 1e-12  # added inside log() to avoid log(0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_spectrogram(
    iq_complex: np.ndarray,
    sampling_freq: float = DEFAULT_SAMPLING_FREQ,
    decimation: int = DEFAULT_DECIMATION,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    window_type: str = DEFAULT_WINDOW_TYPE,
    noverlap: int = DEFAULT_NOVERLAP,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    colormap: str = DEFAULT_COLORMAP,
    log_compress: bool = True,
) -> np.ndarray:
    """
    Convert one file's worth of I/Q samples into a 256x256x3 RGB spectrogram.

    Args:
        iq_complex: complex128 array, output of parse_iq_file().
        sampling_freq: Pre-decimation sample rate (Hz).
        decimation: Descending sampling coefficient.
        window_length: STFT window length (= number of frequency bins).
        window_type: Window function name ('hann' is the default).
        noverlap: Hop = window_length - noverlap. 0 = no overlap.
        output_size: Final spatial size (H = W = output_size).
        colormap: matplotlib colormap name. 'jet' per Pan et al. Appendix A.
        log_compress: Apply 10*log10 to the power. Standard practice.

    Returns:
        uint8 array of shape (output_size, output_size, 3). Values in [0, 255].
    """
    # ---- Step 1: STFT ----
    fs_effective = sampling_freq / decimation
    _, _, Zxx = scipy.signal.stft(
        iq_complex,
        fs=fs_effective,
        window=window_type,
        nperseg=window_length,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,  # don't extend the signal
        padded=False,   # don't zero-pad to a multiple of nperseg
    )
    # Zxx shape: (window_length, n_time_windows)

    # ---- Step 2: fftshift to monotonic frequency axis ----
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    # ---- Step 3: Power spectrum ----
    power = np.abs(Zxx_shifted) ** 2

    # ---- Step 4: Log compression ----
    if log_compress:
        spectrogram = 10.0 * np.log10(power + LOG_EPSILON)
    else:
        spectrogram = power

    # ---- Step 5: Per-file min-max normalisation to [0, 1] ----
    s_min = spectrogram.min()
    s_max = spectrogram.max()
    if s_max - s_min < 1e-12:
        # Degenerate constant-valued spectrogram — return a zero image.
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)
    spectrogram_norm = (spectrogram - s_min) / (s_max - s_min)

    # ---- Step 6: Resize to (output_size, output_size) ----
    # PIL expects uint8 or grayscale float. We round-trip through uint8 so
    # the resize matches what most image libraries do for grayscale.
    img_uint8 = (spectrogram_norm * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_resized_pil = img_pil.resize((output_size, output_size), Image.BILINEAR)
    img_resized = np.array(img_resized_pil) / 255.0  # back to [0, 1]

    # ---- Step 7: Apply Jet colormap → RGB ----
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(img_resized)             # (H, W, 4) float in [0, 1]
    rgb = rgba[:, :, :3]                 # drop alpha
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8


def compute_stft_raw(
    iq_complex: np.ndarray,
    sampling_freq: float = DEFAULT_SAMPLING_FREQ,
    decimation: int = DEFAULT_DECIMATION,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    window_type: str = DEFAULT_WINDOW_TYPE,
    noverlap: int = DEFAULT_NOVERLAP,
    center_freq_hz: float = 99e6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run STFT and return raw power (no log, no normalisation, no colormap).

    Useful for diagnostics: lets you inspect the actual frequency content
    in MHz before any visual transformation.

    Args:
        iq_complex: complex128 array.
        sampling_freq, decimation, window_length, window_type, noverlap:
            Same as iq_to_spectrogram.
        center_freq_hz: RF centre frequency for absolute MHz calculation.

    Returns:
        Tuple of (frequencies_mhz, times_seconds, power):
            frequencies_mhz: shape (window_length,), absolute frequency per bin.
            times_seconds:   shape (n_time_windows,), centre time per window.
            power:           shape (window_length, n_time_windows), linear.
    """
    fs_effective = sampling_freq / decimation
    f, t, Zxx = scipy.signal.stft(
        iq_complex,
        fs=fs_effective,
        window=window_type,
        nperseg=window_length,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    # Reorder so frequencies are monotonically increasing
    f_shifted = np.fft.fftshift(f)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    # Convert to absolute MHz
    f_absolute_mhz = (f_shifted + center_freq_hz) / 1e6

    power = np.abs(Zxx_shifted) ** 2
    return f_absolute_mhz, t, power