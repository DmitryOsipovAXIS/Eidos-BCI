"""Visualize collected SSVEP training data (v2).

Usage:
  python scripts/visualize_data_v2.py
  python scripts/visualize_data_v2.py --file data/raw/X_20240518_143022.npy
  python scripts/visualize_data_v2.py --left-hz 10 --right-hz 15 --sample-rate 125
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig
from signal_processing.filters import bandpass_filter


def load_latest(data_dir: Path):
    x_files = sorted(data_dir.glob("X_*.npy"))
    if not x_files:
        raise FileNotFoundError(f"No X_*.npy files found in {data_dir}")
    x_path = x_files[-1]
    y_path = x_path.parent / x_path.name.replace("X_", "y_")
    return np.load(x_path), np.load(y_path), x_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to X_*.npy file")
    parser.add_argument("--left-hz", type=float, default=None, help="Left target frequency")
    parser.add_argument("--right-hz", type=float, default=None, help="Right target frequency")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sampling rate in Hz")
    parser.add_argument("--record-seconds", type=float, default=30.0,
                        help="Total recording seconds per trial (default 30.0)")
    parser.add_argument("--discard-seconds", type=float, default=0.5,
                        help="Discard seconds at trial start (default 0.5)")
    parser.add_argument("--bandpass-low", type=float, default=None, help="Bandpass low cutoff")
    parser.add_argument("--bandpass-high", type=float, default=None, help="Bandpass high cutoff")
    parser.add_argument("--no-filter", action="store_true", help="Skip bandpass filtering")
    parser.add_argument("--nperseg", type=int, default=256, help="Welch segment length")
    parser.add_argument("--show-trial", type=int, default=0, help="Trial index to preview")
    return parser.parse_args()


def summarize_quality(X: np.ndarray) -> None:
    finite = np.isfinite(X).all()
    zero_frac = float(np.mean(X == 0.0))
    print(f"Finite values: {finite}")
    print(f"Zero fraction: {zero_frac:0.4f}")
    print(f"Value range: min={X.min():0.3f} max={X.max():0.3f}")


def main() -> None:
    args = parse_args()

    config = SSVEPConfig()
    if args.file:
        x_path = Path(args.file)
        y_path = x_path.parent / x_path.name.replace("X_", "y_")
        X = np.load(x_path)
        y = np.load(y_path)
        fname = x_path.name
    else:
        X, y, fname = load_latest(config.data_raw_dir)

    # Defaults aligned with collect_training_data_v2.py
    left_hz = float(args.left_hz) if args.left_hz is not None else 10.0
    right_hz = float(args.right_hz) if args.right_hz is not None else 15.0

    n_trials, n_ch, n_samples = X.shape
    if args.sample_rate is not None:
        fs = float(args.sample_rate)
        fs_note = "(manual)"
    else:
        kept_s = max(1e-6, args.record_seconds - args.discard_seconds)
        fs = n_samples / kept_s
        fs_note = "(inferred from samples)"

        nyq_ok = fs >= (2.0 * max(left_hz, right_hz) + 1.0)
        if not nyq_ok and args.record_seconds != 4.0:
            fallback_s = max(1e-6, 4.0 - args.discard_seconds)
            fallback_fs = n_samples / fallback_s
            if fallback_fs >= (2.0 * max(left_hz, right_hz) + 1.0):
                fs = fallback_fs
                fs_note = "(auto: assumed 4s trials)"

    if args.bandpass_low is not None and args.bandpass_high is not None:
        low, high = float(args.bandpass_low), float(args.bandpass_high)
    else:
        low = max(min(left_hz, right_hz) - config.bandpass_margin_hz, config.bandpass_low_hz)
        high = min(max(left_hz, right_hz) + config.bandpass_margin_hz, config.bandpass_high_hz)

    nyq = 0.5 * fs
    if high >= nyq:
        high = max(low + 0.1, 0.95 * nyq)
        print(f"Adjusted bandpass high to {high:0.2f} Hz to respect Nyquist")

    print(f"File : {fname}")
    print(f"X    : {X.shape}  (trials, channels, samples)")
    print(f"y    : {y.shape}  LEFT={int((y==0).sum())}  RIGHT={int((y==1).sum())}")
    print(f"fs   : {fs:0.2f} Hz {fs_note}")
    summarize_quality(X)
    t = np.arange(n_samples) / fs

    left_trials = X[y == 0]
    right_trials = X[y == 1]

    def maybe_filter(trials: np.ndarray) -> np.ndarray:
        if args.no_filter:
            return trials
        filtered = []
        for trial in trials:
            filtered.append(bandpass_filter(trial, fs, low, high))
        return np.stack(filtered, axis=0)

    left_f = maybe_filter(left_trials)
    right_f = maybe_filter(right_trials)

    # ------------------------------------------------------------------
    # Figure 1: PSD per channel (LEFT vs RIGHT)
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(n_ch, 2, figsize=(12, n_ch * 2.2), sharex=True, sharey=True)
    fig1.suptitle("PSD per channel (LEFT vs RIGHT)", fontsize=13)

    nperseg = min(args.nperseg, n_samples)

    for ci in range(n_ch):
        for col, (trials, label, color) in enumerate([
            (left_f, f"LEFT ({left_hz} Hz)", "steelblue"),
            (right_f, f"RIGHT ({right_hz} Hz)", "tomato"),
        ]):
            ax = axes[ci, col] if n_ch > 1 else axes[col]
            if trials.size == 0:
                ax.set_visible(False)
                continue
            psds = []
            for trial in trials:
                f, pxx = welch(trial[ci], fs=fs, nperseg=nperseg)
                psds.append(pxx)
            psds = np.array(psds)
            mean_psd = np.mean(psds, axis=0)
            std_psd = np.std(psds, axis=0)

            ax.semilogy(f, mean_psd, color=color, linewidth=1.6)
            ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd,
                            alpha=0.2, color=color)
            ax.axvline(left_hz, color="steelblue", linestyle="--", alpha=0.7)
            ax.axvline(right_hz, color="tomato", linestyle="--", alpha=0.7)
            if ci == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"Ch {ci}")
            if ci == n_ch - 1:
                ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim(2, max(20, right_hz + 5))
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: Mean time-domain response per channel
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_ch, 2, figsize=(12, n_ch * 1.8), sharex=True, sharey=True)
    fig2.suptitle("Mean time-domain response (LEFT vs RIGHT)", fontsize=13)

    for ci in range(n_ch):
        for col, (trials, label, color) in enumerate([
            (left_f, "LEFT", "steelblue"),
            (right_f, "RIGHT", "tomato"),
        ]):
            ax = axes2[ci, col] if n_ch > 1 else axes2[col]
            if trials.size == 0:
                ax.set_visible(False)
                continue
            mean_ts = np.mean(trials[:, ci, :], axis=0)
            std_ts = np.std(trials[:, ci, :], axis=0)
            ax.plot(t, mean_ts, color=color, linewidth=1.0)
            ax.fill_between(t, mean_ts - std_ts, mean_ts + std_ts, color=color, alpha=0.2)
            if ci == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"Ch {ci}")
            if ci == n_ch - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3: Single-trial preview (raw vs filtered)
    # ------------------------------------------------------------------
    trial_idx = max(0, min(args.show_trial, n_trials - 1))
    raw_trial = X[trial_idx]
    filt_trial = bandpass_filter(raw_trial, fs, low, high) if not args.no_filter else raw_trial

    fig3, axes3 = plt.subplots(n_ch, 2, figsize=(12, n_ch * 1.6), sharex=True, sharey=True)
    fig3.suptitle(f"Trial {trial_idx} preview (raw vs filtered)", fontsize=13)

    for ci in range(n_ch):
        ax_raw = axes3[ci, 0] if n_ch > 1 else axes3[0]
        ax_filt = axes3[ci, 1] if n_ch > 1 else axes3[1]
        ax_raw.plot(t, raw_trial[ci], color="grey", linewidth=0.7)
        ax_filt.plot(t, filt_trial[ci], color="black", linewidth=0.7)
        if ci == 0:
            ax_raw.set_title("Raw")
            ax_filt.set_title("Filtered")
        ax_raw.set_ylabel(f"Ch {ci}")
        if ci == n_ch - 1:
            ax_raw.set_xlabel("Time (s)")
            ax_filt.set_xlabel("Time (s)")
        ax_raw.grid(True, alpha=0.2)
        ax_filt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
