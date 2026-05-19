"""Visualize collected SSVEP training data.

Usage:
  python scripts/visualize_data.py
  python scripts/visualize_data.py --file data/raw/X_20240518_143022.npy
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to X_*.npy file")
    args = parser.parse_args()

    config = SSVEPConfig()
    fs = config.sample_rate_hz

    if args.file:
        x_path = Path(args.file)
        y_path = x_path.parent / x_path.name.replace("X_", "y_")
        X = np.load(x_path)
        y = np.load(y_path)
        fname = x_path.name
    else:
        X, y, fname = load_latest(config.data_raw_dir)

    print(f"File : {fname}")
    print(f"X    : {X.shape}  (trials, channels, samples)")
    print(f"y    : {y.shape}  LEFT={int((y==0).sum())}  RIGHT={int((y==1).sum())}")

    low, high = config.bandpass_band_hz()
    n_trials, n_ch, n_samples = X.shape
    t = np.arange(n_samples) / fs

    left_trials  = X[y == 0]
    right_trials = X[y == 1]

    # ------------------------------------------------------------------
    # Figure 1: average PSD — LEFT vs RIGHT
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig1.suptitle("Average PSD across trials — LEFT vs RIGHT", fontsize=13)

    for ax, trials, label, color in zip(
        axes,
        [left_trials, right_trials],
        [f"LEFT ({config.left_hz} Hz)", f"RIGHT ({config.right_hz} Hz)"],
        ["steelblue", "tomato"],
    ):
        psds = []
        for trial in trials:
            filtered = bandpass_filter(trial, fs, low, high)
            # average across channels
            ch_psds = []
            for ch in range(n_ch):
                f, pxx = welch(filtered[ch], fs=fs, nperseg=min(256, n_samples))
                ch_psds.append(pxx)
            psds.append(np.mean(ch_psds, axis=0))

        mean_psd = np.mean(psds, axis=0)
        std_psd  = np.std(psds,  axis=0)

        ax.semilogy(f, mean_psd, color=color, linewidth=2, label=label)
        ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd,
                        alpha=0.2, color=color)
        ax.axvline(config.left_hz,  color="steelblue", linestyle="--",
                   alpha=0.7, label=f"{config.left_hz} Hz (left target)")
        ax.axvline(config.right_hz, color="tomato",    linestyle="--",
                   alpha=0.7, label=f"{config.right_hz} Hz (right target)")
        ax.set_xlim(2, 20)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (log scale)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: raw EEG time series — one LEFT and one RIGHT trial
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_ch, 2, figsize=(14, n_ch * 1.4), sharex=True)
    fig2.suptitle("Raw EEG — example LEFT trial (left) vs RIGHT trial (right)", fontsize=13)

    ch_names = ["POz", "O1", "Oz", "O2"]

    for ci in range(n_ch):
        for col, (trial, color) in enumerate(zip(
            [left_trials[0], right_trials[0]],
            ["steelblue", "tomato"]
        )):
            ax = axes2[ci, col]
            ax.plot(t, trial[ci], color=color, linewidth=0.6)
            ax.set_ylabel(ch_names[ci] if ci < len(ch_names) else f"Ch{ci}",
                          fontsize=8)
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)
            if ci == 0:
                ax.set_title("LEFT trial" if col == 0 else "RIGHT trial")
            if ci == n_ch - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3: feature scatter — log band-power at 10 Hz vs 15 Hz
    # ------------------------------------------------------------------
    from features.frequency_features import extract_ssvep_feature_vector

    feats = []
    for trial in X:
        filtered = bandpass_filter(trial, fs, low, high)
        fv = extract_ssvep_feature_vector(filtered, fs, (config.left_hz, config.right_hz))
        feats.append(fv)
    feats = np.array(feats)

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.scatter(feats[y==0, 0], feats[y==0, 1],
                color="steelblue", label="LEFT",  alpha=0.7, s=60)
    ax3.scatter(feats[y==1, 0], feats[y==1, 1],
                color="tomato",    label="RIGHT", alpha=0.7, s=60)
    ax3.set_xlabel(f"log power @ {config.left_hz} Hz")
    ax3.set_ylabel(f"log power @ {config.right_hz} Hz")
    ax3.set_title("Feature space — are LEFT and RIGHT separable?")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
