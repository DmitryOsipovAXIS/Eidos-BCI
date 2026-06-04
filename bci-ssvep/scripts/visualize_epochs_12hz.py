"""Split the 12HzGOED recording into 4s epochs and plot PSD per epoch.

Usage:
  python scripts/visualize_epochs_12hz.py
  python scripts/visualize_epochs_12hz.py --epoch-s 4 --max-epochs 20
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

TARGET_HZ = 12.0
FS = 125.0
DATA_FILE = ROOT / "data" / "raw" / "X_12HzGOED.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=str(DATA_FILE))
    p.add_argument("--epoch-s", type=float, default=4.0, help="Epoch length in seconds")
    p.add_argument("--fs", type=float, default=FS)
    p.add_argument("--target-hz", type=float, default=TARGET_HZ)
    p.add_argument("--max-epochs", type=int, default=None, help="Limit number of epochs shown")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X = np.load(args.file)
    # X shape: (1, n_channels, n_samples) — squeeze the trial dim
    eeg = X[0]  # (n_channels, n_samples)
    n_ch, n_samples = eeg.shape
    epoch_len = int(args.epoch_s * args.fs)

    n_epochs = n_samples // epoch_len
    if args.max_epochs is not None:
        n_epochs = min(n_epochs, args.max_epochs)

    print(f"File       : {args.file}")
    print(f"EEG shape  : {eeg.shape}  (channels, samples)")
    print(f"fs         : {args.fs} Hz")
    print(f"Epoch len  : {epoch_len} samples ({args.epoch_s}s)")
    print(f"Epochs     : {n_epochs}")

    nperseg = epoch_len  # one Welch segment = full epoch -> max freq resolution

    # Compute PSD for each epoch, averaged over channels
    all_freqs = None
    epoch_psds = []
    for i in range(n_epochs):
        epoch = eeg[:, i * epoch_len : (i + 1) * epoch_len]
        ch_psds = []
        for ch in range(n_ch):
            f, pxx = welch(epoch[ch], fs=args.fs, nperseg=nperseg)
            ch_psds.append(pxx)
        epoch_psds.append(np.mean(ch_psds, axis=0))
        all_freqs = f

    epoch_psds = np.array(epoch_psds)  # (n_epochs, n_freqs)

    # --- Figure 1: one subplot per epoch ---
    cols = 5
    rows = (n_epochs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8), sharey=True)
    axes_flat = axes.flatten() if n_epochs > 1 else [axes]
    fig.suptitle(f"PSD per 4s epoch — 12 Hz recording ({n_epochs} epochs, mean over {n_ch} channels)", fontsize=12)

    freq_mask = all_freqs <= 40
    for i in range(n_epochs):
        ax = axes_flat[i]
        ax.semilogy(all_freqs[freq_mask], epoch_psds[i][freq_mask], color="tomato", linewidth=1.2)
        ax.axvline(args.target_hz, color="black", linestyle="--", linewidth=1, alpha=0.7)
        harmonics = [args.target_hz * k for k in range(2, 5) if args.target_hz * k <= 40]
        for h in harmonics:
            ax.axvline(h, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Epoch {i+1}", fontsize=9)
        ax.set_xlabel("Hz", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_epochs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    # --- Figure 2: mean PSD across all epochs with std band ---
    mean_psd = np.mean(epoch_psds, axis=0)
    std_psd = np.std(epoch_psds, axis=0)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.semilogy(all_freqs[freq_mask], mean_psd[freq_mask], color="tomato", linewidth=2, label="Mean PSD")
    ax2.fill_between(
        all_freqs[freq_mask],
        np.maximum(mean_psd[freq_mask] - std_psd[freq_mask], 1e-20),
        mean_psd[freq_mask] + std_psd[freq_mask],
        alpha=0.25, color="tomato", label="±1 std"
    )
    ax2.axvline(args.target_hz, color="black", linestyle="--", linewidth=1.5, label=f"{args.target_hz} Hz")
    for h in [args.target_hz * k for k in range(2, 5) if args.target_hz * k <= 40]:
        ax2.axvline(h, color="grey", linestyle=":", linewidth=1, alpha=0.7, label=f"{h:.0f} Hz harmonic")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_title(f"Mean PSD across {n_epochs} epochs (mean over {n_ch} channels)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
