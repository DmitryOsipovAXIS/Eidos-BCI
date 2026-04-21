"""Frequency-domain features for SSVEP (Welch PSD, band power, simple argmax)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import welch


def welch_psd(
    eeg: np.ndarray,
    sample_rate_hz: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average Welch PSD across EEG channels.

    Args:
        eeg: Shape ``(n_channels, n_samples)``.
        sample_rate_hz: Sampling rate.
        nperseg: Segment length for Welch (default ``min(256, n_samples)``).

    Returns:
        ``(freqs, psd_avg)`` where ``psd_avg`` is 1D over ``freqs``.
    """
    nperseg = nperseg or max(8, min(256, eeg.shape[1]))
    nperseg = min(nperseg, eeg.shape[1])

    psds: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    for ch in range(eeg.shape[0]):
        freqs, pxx = welch(eeg[ch], fs=sample_rate_hz, nperseg=nperseg, axis=-1)
        psds.append(pxx)
        freqs_ref = freqs

    assert freqs_ref is not None
    psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
    return freqs_ref, psd_mean


def band_power_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    center_hz: float,
    half_width_hz: float,
) -> float:
    """Approximate band power from Welch PSD (uniform frequency grid from ``welch``)."""
    band = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    if not np.any(band):
        return 0.0
    if freqs.size < 2:
        return float(np.sum(psd[band]))
    df = float(freqs[1] - freqs[0])
    return float(np.sum(psd[band]) * df)


def dominant_frequency(freqs: np.ndarray, psd: np.ndarray, max_hz: float = 40.0) -> float:
    """Return frequency (Hz) at maximum PSD below ``max_hz`` (coarse SSVEP detector)."""
    mask = freqs <= max_hz
    if not np.any(mask):
        return float(freqs[np.argmax(psd)])
    idx = np.argmax(psd[mask])
    return float(freqs[mask][idx])


def extract_ssvep_feature_vector(
    eeg: np.ndarray,
    sample_rate_hz: float,
    target_frequencies_hz: Sequence[float],
    band_half_width_hz: float = 0.5,
    nperseg: int | None = None,
) -> np.ndarray:
    """Log band-powers at each target frequency (mean across channels in PSD domain).

    Returns:
        1D feature vector of length ``len(target_frequencies_hz)``.
    """
    freqs, psd = welch_psd(eeg, sample_rate_hz, nperseg=nperseg)
    feats = [
        np.log(band_power_from_psd(freqs, psd, f0, band_half_width_hz) + 1e-12)
        for f0 in target_frequencies_hz
    ]
    return np.array(feats, dtype=np.float64)
