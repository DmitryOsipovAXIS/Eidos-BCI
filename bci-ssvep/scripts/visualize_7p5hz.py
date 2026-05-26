"""High-resolution 7.5 Hz visualizer for single-trial SSVEP recordings.

Usage:
    uv run .\\scripts\\visualize_7p5hz.py
    uv run .\\scripts\\visualize_7p5hz.py --file .\\data\\raw\\X_LEFT7.5.npy

The script is optimized for long single-trial recordings where a coarse
Welch window (for example 256 samples) hides the 7.5 Hz peak. It uses a
full-length periodogram for frequency resolution and zooms into the 4-12 Hz
range so the target response is easier to inspect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, periodogram, welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig


def load_pair(x_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    y_path = x_path.parent / x_path.name.replace("X_", "y_")
    return np.load(x_path), np.load(y_path), x_path.name


def infer_sample_rate(n_samples: int, record_seconds: float, discard_seconds: float) -> float:
    kept_seconds = max(1e-6, record_seconds - discard_seconds)
    return n_samples / kept_seconds


def bandpass(eeg: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.5, min(low, nyq * 0.95))
    high = min(high, nyq * 0.99)
    if high <= low:
        return eeg
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to X_*.npy file")
    parser.add_argument("--left-hz", type=float, default=7.5, help="Target frequency to highlight")
    parser.add_argument("--sample-rate", type=float, default=None, help="Override sample rate")
    parser.add_argument("--record-seconds", type=float, default=30.0, help="Nominal recording duration")
    parser.add_argument("--discard-seconds", type=float, default=0.5, help="Discarded start seconds")
    parser.add_argument("--low-hz", type=float, default=4.0, help="Zoom lower frequency")
    parser.add_argument("--high-hz", type=float, default=12.0, help="Zoom upper frequency")
    parser.add_argument("--no-filter", action="store_true", help="Skip bandpass filtering")
    parser.add_argument("--show-channel", type=int, default=None,
                        help="Optional channel index to overlay on the PSD plot")
    args = parser.parse_args()

    config = SSVEPConfig()
    if args.file:
        x_path = Path(args.file)
        X, y, fname = load_pair(x_path)
    else:
        x_files = sorted(config.data_raw_dir.glob("X_*.npy"))
        if not x_files:
            raise FileNotFoundError(f"No X_*.npy files found in {config.data_raw_dir}")
        X, y, fname = load_pair(x_files[-1])

    n_trials, n_ch, n_samples = X.shape
    fs = float(args.sample_rate) if args.sample_rate is not None else infer_sample_rate(
        n_samples, args.record_seconds, args.discard_seconds
    )

    print(f"File : {fname}")
    print(f"X    : {X.shape}  (trials, channels, samples)")
    print(f"y    : {y.shape}  LEFT={int((y==0).sum())}  RIGHT={int((y==1).sum())}")
    print(f"fs   : {fs:.3f} Hz")

    trial = X[0]
    if not args.no_filter:
        trial = bandpass(trial, fs, max(0.5, args.left_hz - 2.0), min(args.high_hz, args.left_hz + 8.0))

    # Use full-length periodogram for fine frequency resolution.
    freqs = None
    psds = []
    for ch in range(n_ch):
        freqs, pxx = periodogram(trial[ch], fs=fs, scaling="density", detrend="constant")
        psds.append(pxx)
    psds = np.asarray(psds)
    mean_psd = psds.mean(axis=0)

    # Welch reference for a smoothed curve.
    welch_freqs = None
    welch_psds = []
    welch_nperseg = min(n_samples, max(1024, int(fs * 8)))
    for ch in range(n_ch):
        welch_freqs, pxx = welch(trial[ch], fs=fs, nperseg=welch_nperseg)
        welch_psds.append(pxx)
    welch_psd = np.mean(np.asarray(welch_psds), axis=0)

    zoom_mask = (freqs >= args.low_hz) & (freqs <= args.high_hz)
    peak_idx = np.argmax(mean_psd[zoom_mask])
    peak_freq = float(freqs[zoom_mask][peak_idx])
    peak_power = float(mean_psd[zoom_mask][peak_idx])
    target_idx = int(np.argmin(np.abs(freqs - args.left_hz)))
    target_power = float(mean_psd[target_idx])

    print(f"Peak in {args.low_hz}-{args.high_hz} Hz window: {peak_freq:.3f} Hz")
    print(f"Power at {args.left_hz:.2f} Hz bin ({freqs[target_idx]:.3f} Hz): {target_power:.3g}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"7.5 Hz visual check — {fname}", fontsize=15)

    # Raw/filtered time series for the first channel.
    t = np.arange(n_samples) / fs
    ax = axes[0, 0]
    ax.plot(t, X[0, 0], color="0.65", linewidth=0.8, label="raw")
    if not args.no_filter:
        ax.plot(t, trial[0], color="black", linewidth=0.9, label="filtered")
    ax.set_title("Channel 0 time series")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # High-resolution PSD zoom.
    ax = axes[0, 1]
    ax.plot(freqs[zoom_mask], mean_psd[zoom_mask], color="steelblue", linewidth=1.8, label="periodogram mean")
    ax.plot(welch_freqs, welch_psd, color="tomato", alpha=0.75, linewidth=1.0,
            label=f"Welch (nperseg={welch_nperseg})")
    ax.axvline(args.left_hz, color="seagreen", linestyle="--", linewidth=1.5,
               label=f"target {args.left_hz:.1f} Hz")
    ax.axvline(2 * args.left_hz, color="orange", linestyle=":", linewidth=1.2,
               label=f"2nd harmonic {2 * args.left_hz:.1f} Hz")
    ax.set_xlim(args.low_hz, args.high_hz)
    ax.set_yscale("log")
    ax.set_title("Zoomed PSD (log scale)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Channel comparison in the zoom window.
    ax = axes[1, 0]
    for ch in range(n_ch):
        _, ch_psd = periodogram(trial[ch], fs=fs, scaling="density", detrend="constant")
        ax.plot(freqs[zoom_mask], ch_psd[zoom_mask], linewidth=0.9, alpha=0.7, label=f"ch {ch}")
    ax.axvline(args.left_hz, color="seagreen", linestyle="--", linewidth=1.3)
    ax.set_title("Per-channel PSD around target")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")
    ax.set_xlim(args.low_hz, args.high_hz)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bar view of target vs nearby frequencies.
    ax = axes[1, 1]
    test_freqs = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 15.0]
    values = []
    for target in test_freqs:
        idx = int(np.argmin(np.abs(freqs - target)))
        values.append(float(mean_psd[idx]))
    bars = ax.bar([str(f) for f in test_freqs], values, color=["#7aa6ff" if f == args.left_hz else "#cfd8dc" for f in test_freqs])
    ax.set_title("Target-bin comparison")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, freq in zip(bars, test_freqs):
        if freq == args.left_hz:
            bar.set_color("#2e86de")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()