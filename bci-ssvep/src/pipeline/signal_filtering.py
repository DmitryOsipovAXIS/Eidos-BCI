"""Sequential signal filtering for SSVEP: notch (50 Hz) then bandpass (3-30 Hz).

Processing order (per expert recommendation):
  1. Notch at 50 Hz  — removes European AC interference
  2. Butterworth bandpass 3-30 Hz  — keeps SSVEP band, discards drift & HF noise

The bandpass upper limit of 30 Hz covers up to 3 harmonics of an 8 Hz stimulus
(8, 16, 24 Hz) and 2 harmonics of a 12 Hz stimulus (12, 24 Hz), while staying
well within the Nyquist limit of 62.5 Hz at 125 Hz sampling.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def notch_filter(
    data: np.ndarray,
    sample_rate_hz: float,
    notch_hz: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Remove a narrow frequency band (default 50 Hz AC line noise).

    Args:
        data: Shape ``(..., n_samples)``; filtered along the last axis.
        sample_rate_hz: Sampling frequency in Hz.
        notch_hz: Centre frequency to notch out (Hz).
        quality_factor: Q-factor controlling notch width. Higher = narrower.

    Returns:
        Filtered array with the same shape as ``data``.
    """
    b, a = iirnotch(notch_hz, quality_factor, fs=sample_rate_hz)
    return filtfilt(b, a, data, axis=-1)


def bandpass_filter(
    data: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 3.0,
    high_hz: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        data: Shape ``(..., n_samples)``; filtered along the last axis.
        sample_rate_hz: Sampling frequency in Hz.
        low_hz: Low cutoff (Hz). Must be > 0.
        high_hz: High cutoff (Hz). Capped at 0.99 * Nyquist.
        order: Butterworth filter order.

    Returns:
        Filtered array with the same shape as ``data``.
    """
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError(f"Require 0 < low_hz < high_hz, got {low_hz}, {high_hz}")
    nyq = 0.5 * sample_rate_hz
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        raise ValueError("Invalid band after Nyquist normalisation")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def apply_filters(
    data: np.ndarray,
    sample_rate_hz: float = 125.0,
    notch_hz: float = 50.0,
    bandpass_low_hz: float = 3.0,
    bandpass_high_hz: float = 30.0,
) -> np.ndarray:
    """Apply the full SSVEP filter chain: notch → bandpass.

    Args:
        data: EEG array ``(n_channels, n_samples)`` or ``(..., n_samples)``.
        sample_rate_hz: Sampling frequency in Hz.
        notch_hz: AC line frequency to remove (50 Hz Europe, 60 Hz North America).
        bandpass_low_hz: Bandpass lower cutoff (Hz).
        bandpass_high_hz: Bandpass upper cutoff (Hz).

    Returns:
        Filtered array with the same shape as ``data``.
    """
    data = notch_filter(data, sample_rate_hz, notch_hz)
    data = bandpass_filter(data, sample_rate_hz, bandpass_low_hz, bandpass_high_hz)
    return data
