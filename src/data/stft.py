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
# scipy.signal is the SciPy submodule for signal processing.
# Provides STFT, filters, convolutions, peak-finding, etc.
# We use it specifically for scipy.signal.stft which handles the windowing
# and FFT chopping logic for us in one call.

from PIL import Image
# PIL (Python Imaging Library, modern fork: Pillow) handles image I/O and
# basic image transformations. We use it here only for the resize step.


# ─────────────────────────────────────────────────────────────────────────────
# Default STFT configuration (Pan et al. Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

# These are the parameters Pan et al. used; we copy them verbatim for reproduction.
DEFAULT_SAMPLING_FREQ = 125_000_000  # Hz, before decimation
# The SDR samples the radio at 125 megasamples per second (very fast).
# This is the raw rate before any downsampling.

DEFAULT_DECIMATION = 4
# Decimation = throw away samples to reduce data rate.
# A decimation of 4 means "keep every 4th sample, discard the others."
# Pan et al. configured the SDR with decimation=4, so the effective sample
# rate is 125 MHz / 4 = 31.25 MHz.
# Why decimate? Because the FM band only spans about 88-108 MHz (20 MHz wide),
# so you don't need 125 MHz of bandwidth — 31.25 MHz is plenty and saves disk.

DEFAULT_WINDOW_LENGTH = 256
# Each STFT window covers 256 samples.
# At 31.25 MHz sample rate, this is 256 / 31.25e6 = ~8.2 microseconds per window.
# This is the TIME RESOLUTION: every column of the spectrogram represents
# about 8.2 microseconds of signal.
# 256 is also the number of FREQUENCY BINS we'll get out, because FFT output
# size = input size. So our 256-pixel-tall spectrogram has 256 frequency bins.

DEFAULT_WINDOW_TYPE = 'hann'
# Hann window — see the windowing explanation above.

DEFAULT_NOVERLAP = 0
# Adjacent STFT windows don't overlap; each window is independent.
# With overlap=0, you get more independent time windows but lose some
# smoothness. Pan et al. specified noverlap=0 in their paper.
# 32508 samples / 256 per window = ~127, so we get 126 time windows.

DEFAULT_OUTPUT_SIZE = 256  # H = W = 256 per paper
DEFAULT_COLORMAP = 'jet'
LOG_EPSILON = 1e-12
# 1e-12 = 0.000000000001 = a very tiny positive number.
# Added inside log() to prevent log(0) = -infinity, which would break math.
# At any pixel where power is genuinely zero, log(0 + epsilon) = log(1e-12) = -27.6.
# A finite, very-negative number — represents "no signal" without blowing up.


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
    # The effective sample rate AFTER decimation.
    # This is what determines what frequencies the STFT can represent (Nyquist).
    fs_effective = sampling_freq / decimation  # = 31.25 MHz

    # scipy.signal.stft returns three things:
    #   f: array of frequencies (Hz), one per FFT bin — shape (window_length,)
    #   t: array of time centres for each window (seconds) — shape (n_windows,)
    #   Zxx: the complex STFT itself — shape (window_length, n_windows)
    # We only use Zxx here; we discard f and t with the underscores.
    _, _, Zxx = scipy.signal.stft(
        iq_complex,
        fs=fs_effective,
        window=window_type,
        nperseg=window_length,
        noverlap=noverlap,

        # CRITICAL: return_onesided=False.
        # By default, scipy.signal.stft assumes a REAL signal and returns only
        # the positive frequencies (because for real signals, negative frequencies
        # are just mirror copies of positive ones). But our signal is COMPLEX
        # baseband I/Q — the positive and negative frequencies carry independent
        # information. So we need the full two-sided spectrum.
        return_onesided=False,

        boundary=None,  # don't pad the signal with extra zeros at the start/end
        padded=False,   # don't zero-pad to a multiple of nperseg
    )
    # Zxx shape: (window_length, n_time_windows) = (256, 126) for FM
    # Each entry Zxx[freq_bin, time_window] is a complex number representing
    # "the strength and phase of frequency bin `freq_bin` during time window `time_window`."

    # ---- Step 2: fftshift to monotonic frequency axis ----
    # FFT output has a weird ordering: it returns
    #   bin 0 = DC (zero frequency)
    #   bins 1 to N/2-1 = positive frequencies (0 to Nyquist)
    #   bins N/2 to N-1 = negative frequencies (-Nyquist to 0)
    #
    # So a raw FFT output looks like:
    #   [DC, +1Hz, +2Hz, ..., +Nyquist-1Hz, -Nyquist, ..., -2Hz, -1Hz]
    #
    # fftshift swaps the two halves so frequencies increase monotonically:
    #   [-Nyquist, ..., -1Hz, DC, +1Hz, ..., +Nyquist-1Hz]
    #
    # axes=0 tells fftshift to operate on axis 0 (the frequency axis).
    # We don't shift axis 1 (time) because time is already in order.
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    # ---- Step 3: Power spectrum ----
    # |Zxx|² is the POWER at each (freq, time) cell.
    # np.abs() on a complex number returns sqrt(real² + imag²) — the magnitude.
    # Squaring gives power (proportional to energy per unit time).
    # This is a real, non-negative number.
    power = np.abs(Zxx_shifted) ** 2

    # ---- Step 4: Log compression ----
    # Why? Power values in spectrograms can span MANY orders of magnitude.
    # An FM transmitter might produce power values like 1e10, while background
    # noise is more like 1e2 or smaller. That's a 100,000,000x range.
    #
    # If we displayed this linearly, the FM transmitter would be a single
    # bright pixel and everything else would look black. Log compression
    # squashes the range:
    #   power 1e10  → 10·log10(1e10) = 100 dB
    #   power 1e2   → 10·log10(1e2)  = 20 dB
    #
    # Now everything is in a reasonable range and visible together.
    # This is exactly what "decibel" (dB) means — a logarithmic measure of power.
    if log_compress:
        spectrogram = 10.0 * np.log10(power + LOG_EPSILON)
        # The + LOG_EPSILON prevents log(0) = -inf at pixels with literally no signal.
    else:
        spectrogram = power

    # ---- Step 5: Per-file min-max normalisation to [0, 1] ----
    # Different files have different min/max power. To get consistent visualization,
    # we rescale each spectrogram so its minimum becomes 0 and its maximum becomes 1.
    # This is "PER-FILE" normalization, meaning each file is normalized independently.
    #
    # Caveat: this throws away the absolute power level. A loud file and a quiet
    # file end up looking the same. For prediction this is probably fine (relative
    # patterns are what matters), but it does discard information.
    s_min = spectrogram.min()
    s_max = spectrogram.max()
    if s_max - s_min < 1e-12:
        # Edge case: if min and max are equal, the spectrogram is constant.
        # Dividing by zero would give NaN. Return a zero image instead.
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    # Linear rescaling: new = (old - min) / (max - min)
    # This is element-wise — numpy broadcasts the subtraction and division
    # across every cell in the 2D array.
    spectrogram_norm = (spectrogram - s_min) / (s_max - s_min)

    # ---- Step 6: Resize to (output_size, output_size) ----
    # Our spectrogram is currently shape (256, 126) for FM files.
    # We need (256, 256) for the model. So we resize.
    #
    # BILINEAR INTERPOLATION = the standard, smooth way to resize an image.
    # For each output pixel, it computes a weighted average of the 4 nearest
    # input pixels. Smooth and fast, but slightly blurs sharp features.
    #
    # We round-trip through uint8 (0-255) and back to [0, 1] because PIL
    # works best with integer pixel values.
    img_uint8 = (spectrogram_norm * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_resized_pil = img_pil.resize((output_size, output_size), Image.BILINEAR)
    img_resized = np.array(img_resized_pil) / 255.0  # back to [0, 1]

    # ---- Step 7: Apply Jet colormap → RGB ----
    # A COLORMAP is a function that maps a scalar value [0, 1] to a color (RGB).
    # The 'jet' colormap goes blue → cyan → green → yellow → red as the value
    # increases from 0 to 1.
    #
    # Why turn a grayscale value into a color? Two reasons:
    # 1. Human eyes detect color differences better than brightness differences,
    #    so colormapped spectrograms look much more informative.
    # 2. Pan et al. used Jet, and we're reproducing them. (Their model EXPECTS
    #    RGB Jet-colored input — that's what it was trained on.)
    cmap = matplotlib.colormaps[colormap]

    # Applying cmap() to a 2D array returns a (H, W, 4) array — RGBA.
    # The A is alpha (transparency), which we don't need.
    rgba = cmap(img_resized)             # (H, W, 4) float in [0, 1]
    rgb = rgba[:, :, :3]                 # drop alpha; numpy slice keeps first 3 channels
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Final output: (256, 256, 3) uint8 array, values in [0, 255].
    # This is a "regular" RGB image you could save with PIL.Image.fromarray().
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
    # This is a DIAGNOSTIC version that returns the same STFT but with
    # human-meaningful frequency axes (in MHz, absolute frequency).
    # Not used by the model pipeline — only for inspecting the data.

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

    # Same fftshift as before — put frequencies in increasing order.
    f_shifted = np.fft.fftshift(f)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    # ── Convert frequency axis to absolute MHz ──
    # scipy.signal.stft returns frequencies in BASEBAND form — meaning
    # frequencies relative to the center frequency of the radio capture.
    # So f might range from -15.625 MHz to +15.625 MHz.
    # To get ABSOLUTE frequencies (the actual MHz of those radio waves),
    # we add the center frequency back in.
    # 99 MHz is the center; +1 MHz baseband becomes 100 MHz absolute.
    # Divide by 1e6 to convert Hz → MHz for readability.
    f_absolute_mhz = (f_shifted + center_freq_hz) / 1e6

    power = np.abs(Zxx_shifted) ** 2
    return f_absolute_mhz, t, power