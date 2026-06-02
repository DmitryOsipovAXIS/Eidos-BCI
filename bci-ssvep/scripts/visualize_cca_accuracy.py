"""CCA accuracy visualizer — loads raw data, splits into epochs, runs CCA, plots results.

Usage:
  python scripts/visualize_cca_accuracy.py
  python scripts/visualize_cca_accuracy.py --file data/raw/X_20260519_120602.npy
  python scripts/visualize_cca_accuracy.py --left-hz 7.5 --right-hz 12 --epoch-s 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
w
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig
from pipeline.cca_classifier import SSVEPClassifierCCA
from pipeline.preprocessing import common_average_reference
from signal_processing.filters import bandpass_filter


def load_latest(data_dir: Path):
    x_files = sorted(data_dir.glob("X_*.npy"))
    if not x_files:
        raise FileNotFoundError(f"No X_*.npy files found in {data_dir}")
    x_path = x_files[-1]
    y_path = x_path.parent / x_path.name.replace("X_", "y_")
    return np.load(x_path), np.load(y_path), x_path.name


def split_into_epochs(X: np.ndarray, y: np.ndarray, fs: float, epoch_s: float, discard_s: float = 0.5):
    """Split (n_trials, n_ch, n_samples) into sub-epochs of length epoch_s."""
    epoch_samples = int(epoch_s * fs)
    discard_samples = int(discard_s * fs)

    X_epochs, y_epochs = [], []
    for trial_idx in range(X.shape[0]):
        n_samples = X.shape[2]
        for start in range(discard_samples, n_samples - epoch_samples + 1, epoch_samples):
            X_epochs.append(X[trial_idx, :, start:start + epoch_samples])
            y_epochs.append(y[trial_idx])

    return np.stack(X_epochs), np.array(y_epochs, dtype=np.int64)


def preprocess(epoch: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    epoch = common_average_reference(epoch)
    epoch = bandpass_filter(epoch, fs, low, high)
    return epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to X_*.npy file (default: latest in data/raw)")
    parser.add_argument("--left-hz",   type=float, default=None, help="Left stimulus frequency")
    parser.add_argument("--right-hz",  type=float, default=None, help="Right stimulus frequency")
    parser.add_argument("--epoch-s",   type=float, default=2.0,  help="Sub-epoch length in seconds (default 2.0)")
    parser.add_argument("--harmonics", type=int,   default=3,    help="CCA harmonics (default 3)")
    args = parser.parse_args()

    config = SSVEPConfig(
        left_hz=args.left_hz   if args.left_hz  is not None else 7.5,
        right_hz=args.right_hz if args.right_hz is not None else 12.0,
    )
    fs = config.sample_rate_hz
    low, high = config.bandpass_band_hz()

    if args.file:
        x_path = Path(args.file)
        y_path = x_path.parent / x_path.name.replace("X_", "y_")
        X = np.load(x_path)
        y = np.load(y_path)
        fname = x_path.name
    else:
        X, y, fname = load_latest(config.data_raw_dir)

    print(f"File     : {fname}")
    print(f"X shape  : {X.shape}  (trials, channels, samples)")
    print(f"y shape  : {y.shape}  LEFT={int((y==0).sum())}  RIGHT={int((y==1).sum())}")
    print(f"Freqs    : LEFT={config.left_hz} Hz  RIGHT={config.right_hz} Hz")
    print(f"Epoch    : {args.epoch_s} s  ({int(args.epoch_s * fs)} samples)")

    X_ep, y_ep = split_into_epochs(X, y, fs, args.epoch_s)
    print(f"Epochs   : {X_ep.shape[0]} total  LEFT={int((y_ep==0).sum())}  RIGHT={int((y_ep==1).sum())}")

    clf = SSVEPClassifierCCA(
        frequencies_hz=[config.left_hz, config.right_hz],
        sample_rate_hz=fs,
        n_harmonics=args.harmonics,
    )

    y_pred = []
    corrs_left  = []
    corrs_right = []

    for i, epoch in enumerate(X_ep):
        proc = preprocess(epoch, fs, low, high)
        pred_idx, corrs = clf.predict(proc)
        y_pred.append(pred_idx)
        corrs_left.append(corrs[0])
        corrs_right.append(corrs[1])
        if (i + 1) % 10 == 0:
            print(f"  classified {i+1}/{len(X_ep)} epochs...", flush=True)

    y_pred = np.array(y_pred)
    corrs_left  = np.array(corrs_left)
    corrs_right = np.array(corrs_right)

    correct = y_pred == y_ep
    accuracy = float(correct.mean()) * 100.0
    acc_left  = float(correct[y_ep == 0].mean()) * 100.0 if (y_ep == 0).any() else float("nan")
    acc_right = float(correct[y_ep == 1].mean()) * 100.0 if (y_ep == 1).any() else float("nan")

    print(f"\nOverall accuracy : {accuracy:.1f}%")
    print(f"LEFT  accuracy   : {acc_left:.1f}%")
    print(f"RIGHT accuracy   : {acc_right:.1f}%")

    labels_str = np.array(["LEFT", "RIGHT"])

    # ------------------------------------------------------------------
    # Figure 1: predicted vs actual — colour-coded strip
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig1.suptitle(f"CCA — predicted vs actual  |  {fname}  |  accuracy {accuracy:.1f}%", fontsize=12)

    epoch_idx = np.arange(len(y_ep))
    colors_actual = ["steelblue" if v == 0 else "tomato" for v in y_ep]
    colors_pred   = ["steelblue" if v == 0 else "tomato" for v in y_pred]

    for ax, colors, title in zip(axes, [colors_actual, colors_pred], ["Actual", "Predicted"]):
        ax.bar(epoch_idx, np.ones(len(epoch_idx)), color=colors, width=1.0, align="center")
        ax.set_yticks([0.5])
        ax.set_yticklabels([title], fontsize=10)
        ax.set_ylim(0, 1)

        # mark wrong predictions on the predicted row
        if title == "Predicted":
            wrong_idx = epoch_idx[~correct]
            ax.scatter(wrong_idx, np.full(len(wrong_idx), 0.5), marker="x",
                       color="white", s=40, linewidths=1.5, zorder=3, label="Wrong")
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Epoch index")
    left_patch  = mpatches.Patch(color="steelblue", label=f"LEFT  ({config.left_hz} Hz)")
    right_patch = mpatches.Patch(color="tomato",    label=f"RIGHT ({config.right_hz} Hz)")
    fig1.legend(handles=[left_patch, right_patch], loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: CCA correlation scores per epoch
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.suptitle("CCA correlation scores per epoch", fontsize=12)

    ax2.plot(epoch_idx, corrs_left,  color="steelblue", linewidth=1.2,
             label=f"ρ LEFT ({config.left_hz} Hz)", alpha=0.85)
    ax2.plot(epoch_idx, corrs_right, color="tomato", linewidth=1.2,
             label=f"ρ RIGHT ({config.right_hz} Hz)", alpha=0.85)

    # shade background by actual label
    for i, (label, col) in enumerate(zip(y_ep, colors_actual)):
        ax2.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.08)

    # mark wrong epochs
    wrong_idx = epoch_idx[~correct]
    ax2.scatter(wrong_idx, np.maximum(corrs_left[~correct], corrs_right[~correct]) + 0.01,
                marker="v", color="black", s=35, zorder=4, label="Misclassified")

    ax2.set_xlabel("Epoch index")
    ax2.set_ylabel("CCA correlation (ρ)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3: accuracy summary bar chart + per-class breakdown
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 5))
    fig3.suptitle(f"CCA accuracy summary — {fname}", fontsize=12)

    # left: overall + per-class accuracy bars
    cats   = ["Overall", f"LEFT\n({config.left_hz} Hz)", f"RIGHT\n({config.right_hz} Hz)"]
    accs   = [accuracy, acc_left, acc_right]
    colors_bar = ["dimgray", "steelblue", "tomato"]
    bars = ax3a.bar(cats, accs, color=colors_bar, width=0.5, edgecolor="white")
    ax3a.set_ylim(0, 105)
    ax3a.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Chance (50%)")
    ax3a.set_ylabel("Accuracy (%)")
    ax3a.set_title("Accuracy per class")
    ax3a.legend(fontsize=8)
    for bar, acc in zip(bars, accs):
        ax3a.text(bar.get_x() + bar.get_width() / 2, acc + 1.5,
                  f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3a.grid(True, axis="y", alpha=0.25)

    # right: 2x2 confusion matrix style
    n_ll = int(((y_ep == 0) & (y_pred == 0)).sum())  # true LEFT,  pred LEFT
    n_lr = int(((y_ep == 0) & (y_pred == 1)).sum())  # true LEFT,  pred RIGHT
    n_rl = int(((y_ep == 1) & (y_pred == 0)).sum())  # true RIGHT, pred LEFT
    n_rr = int(((y_ep == 1) & (y_pred == 1)).sum())  # true RIGHT, pred RIGHT
    cm = np.array([[n_ll, n_lr], [n_rl, n_rr]])

    im = ax3b.imshow(cm, cmap="Blues", vmin=0)
    ax3b.set_xticks([0, 1]); ax3b.set_xticklabels(["Pred LEFT", "Pred RIGHT"])
    ax3b.set_yticks([0, 1]); ax3b.set_yticklabels(["True LEFT", "True RIGHT"])
    ax3b.set_title("Confusion matrix (epochs)")
    for r in range(2):
        for c in range(2):
            ax3b.text(c, r, str(cm[r, c]), ha="center", va="center",
                      fontsize=14, fontweight="bold",
                      color="white" if cm[r, c] > cm.max() * 0.6 else "black")
    fig3.colorbar(im, ax=ax3b, fraction=0.046, pad=0.04)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()