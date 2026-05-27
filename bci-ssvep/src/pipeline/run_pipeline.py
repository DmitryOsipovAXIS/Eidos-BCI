"""Offline SSVEP pipeline runner.

Loads X/y .npy files from data/raw/, runs the full preprocessing + CCA pipeline,
saves the preprocessed epochs to data/processed/, then visualizes results.

Usage (from project root):
    python src/pipeline/run_pipeline.py
    python src/pipeline/run_pipeline.py --file X_20260519_120602.npy
    python src/pipeline/run_pipeline.py --left-hz 8 --right-hz 15 --harmonics 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from pipeline.pipeline import SSVEPPipeline
from pipeline.signal_filtering import apply_filters
from pipeline.preprocessing import common_average_reference


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_latest_pair(data_dir: Path) -> tuple[Path, Path]:
    x_files = sorted(data_dir.glob("X_*.npy"), key=lambda p: p.stat().st_mtime)
    if not x_files:
        raise FileNotFoundError(f"No X_*.npy files found in {data_dir}")
    x_path = x_files[-1]
    y_path = x_path.parent / x_path.name.replace("X_", "y_", 1)
    if not y_path.exists():
        raise FileNotFoundError(f"Expected label file {y_path} not found")
    return x_path, y_path


def load_pair(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 3:
        raise ValueError(f"X must be (n_trials, n_channels, n_samples), got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"y shape {y.shape} does not match X trials {X.shape[0]}")
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_epochs(
    X: np.ndarray,
    sample_rate_hz: float,
    notch_hz: float,
    bp_low: float,
    bp_high: float,
) -> np.ndarray:
    """Apply CAR + notch + bandpass to every trial. Returns same shape as X."""
    out = np.empty_like(X)
    for i, epoch in enumerate(X):
        epoch = common_average_reference(epoch)
        epoch = apply_filters(epoch, sample_rate_hz, notch_hz, bp_low, bp_high)
        out[i] = epoch
    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _time_axis(n_samples: int, fs: float) -> np.ndarray:
    return np.arange(n_samples) / fs


def visualize(
    X_raw: np.ndarray,
    X_proc: np.ndarray,
    y: np.ndarray,
    results: dict,
    frequencies_hz: list[float],
    sample_rate_hz: float,
    labels: list[str],
    save_dir: Path | None = None,
) -> None:
    """Four-panel figure per class: raw vs filtered time-series, PSD, CCA scores."""
    n_trials, n_ch, n_samples = X_raw.shape
    t = _time_axis(n_samples, sample_rate_hz)
    freqs_fft = np.fft.rfftfreq(n_samples, 1.0 / sample_rate_hz)

    # --- Figure 1: raw vs preprocessed for first trial of each class ---
    fig, axes = plt.subplots(
        len(labels), 2, figsize=(14, 4 * len(labels)), sharex="col"
    )
    if len(labels) == 1:
        axes = axes[np.newaxis, :]

    for cls_i, lbl in enumerate(labels):
        trial_idx = next((i for i, yi in enumerate(y) if yi == cls_i), None)
        if trial_idx is None:
            continue
        raw = X_raw[trial_idx]
        proc = X_proc[trial_idx]
        ax_raw, ax_proc = axes[cls_i]
        for ch in range(n_ch):
            ax_raw.plot(t, raw[ch], alpha=0.7, lw=0.8, label=f"ch{ch}")
            ax_proc.plot(t, proc[ch], alpha=0.7, lw=0.8, label=f"ch{ch}")
        ax_raw.set_title(f"{lbl} — raw (trial {trial_idx + 1})")
        ax_proc.set_title(f"{lbl} — preprocessed")
        ax_raw.set_ylabel("counts")
        ax_proc.set_ylabel("counts")
        if cls_i == len(labels) - 1:
            ax_raw.set_xlabel("Time (s)")
            ax_proc.set_xlabel("Time (s)")

    fig.suptitle("Raw vs Preprocessed EEG (first trial per class)", fontsize=13)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "raw_vs_preprocessed.png", dpi=120)

    # --- Figure 2: PSD averaged over channels, per class ---
    fig2, axes2 = plt.subplots(1, len(labels), figsize=(7 * len(labels), 4), sharey=True)
    if len(labels) == 1:
        axes2 = [axes2]

    for cls_i, lbl in enumerate(labels):
        class_trials = X_proc[y == cls_i]
        if len(class_trials) == 0:
            continue
        # Average PSD across trials and channels
        psds = []
        for epoch in class_trials:
            fft_mag = np.abs(np.fft.rfft(epoch, axis=-1)) ** 2
            psds.append(fft_mag.mean(axis=0))
        mean_psd = np.mean(psds, axis=0)
        ax = axes2[cls_i]
        ax.semilogy(freqs_fft, mean_psd, lw=1.2)
        for f in frequencies_hz:
            ax.axvline(f, color="red", lw=1, ls="--", alpha=0.7)
            for h in range(2, 4):
                ax.axvline(f * h, color="orange", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlim(0, 45)
        ax.set_title(f"PSD — {lbl} (avg {len(class_trials)} trials)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (a.u.)")

    fig2.suptitle("Average PSD per class after preprocessing", fontsize=13)
    fig2.tight_layout()
    if save_dir:
        fig2.savefig(save_dir / "psd_per_class.png", dpi=120)

    # --- Figure 3: CCA scores per trial ---
    cca_scores = np.array(results["cca_scores"])  # (n_trials, n_classes)
    preds = results["predictions"]
    true_labels_str = [labels[int(yi)] for yi in y]

    fig3, ax3 = plt.subplots(figsize=(max(8, n_trials * 0.6), 4))
    x_pos = np.arange(n_trials)
    width = 0.35
    for cls_i, lbl in enumerate(labels):
        ax3.bar(x_pos + cls_i * width, cca_scores[:, cls_i], width, label=f"CCA {lbl}", alpha=0.8)

    # Mark correct / wrong
    for i, (true, pred) in enumerate(zip(true_labels_str, preds)):
        colour = "green" if true == pred else "red"
        ax3.annotate(
            "✓" if true == pred else "✗",
            xy=(i + 0.175, cca_scores[i].max() + 0.01),
            ha="center",
            fontsize=10,
            color=colour,
        )

    ax3.set_xticks(x_pos + width / 2)
    ax3.set_xticklabels([f"T{i+1}\n({labels[int(yi)]})" for i, yi in enumerate(y)], fontsize=8)
    ax3.set_ylabel("CCA correlation")
    ax3.set_title(
        f"CCA scores per trial — accuracy {results['accuracy']:.1%}"
        f" ({results['n_correct']}/{results['n_total']})"
    )
    ax3.legend()
    fig3.tight_layout()
    if save_dir:
        fig3.savefig(save_dir / "cca_scores.png", dpi=120)

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SSVEP CCA pipeline")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--file", default=None, help="Specific X_*.npy filename")
    parser.add_argument("--left-hz", type=float, default=7.5)
    parser.add_argument("--right-hz", type=float, default=12.0)
    parser.add_argument("--sample-rate", type=float, default=125.0)
    parser.add_argument("--harmonics", type=int, default=6)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--bp-low", type=float, default=3.0)
    parser.add_argument("--bp-high", type=float, default=60.0)
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        x_path = data_dir / args.file
        y_path = data_dir / args.file.replace("X_", "y_", 1)
    else:
        x_path, y_path = find_latest_pair(data_dir)

    print(f"Loading  X : {x_path}")
    print(f"         y : {y_path}")
    X_raw, y = load_pair(x_path, y_path)
    n_trials, n_ch, n_samples = X_raw.shape
    print(f"Shape    X : {X_raw.shape}  (trials x channels x samples)")
    print(f"         y : unique={np.unique(y, return_counts=True)}")
    print()

    frequencies_hz = [args.left_hz, args.right_hz]
    labels = ["LEFT", "RIGHT"]

    # Preprocess
    print("Preprocessing (CAR -> notch -> bandpass)...")
    X_proc = preprocess_epochs(
        X_raw,
        sample_rate_hz=args.sample_rate,
        notch_hz=args.notch_hz,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
    )

    # Save preprocessed data
    stem = x_path.stem.replace("X_", "")
    out_x = processed_dir / f"X_proc_{stem}.npy"
    out_y = processed_dir / f"y_proc_{stem}.npy"
    np.save(out_x, X_proc)
    np.save(out_y, y)
    print(f"Saved preprocessed X -> {out_x}")
    print(f"Saved labels       y -> {out_y}")
    print()

    # Classify
    print("Running CCA classifier on each trial...")
    pipeline = SSVEPPipeline(
        frequencies_hz=frequencies_hz,
        sample_rate_hz=args.sample_rate,
        n_harmonics=args.harmonics,
        notch_hz=args.notch_hz,
        bandpass_low_hz=args.bp_low,
        bandpass_high_hz=args.bp_high,
        labels=labels,
    )
    true_labels_str = [labels[int(yi)] for yi in y]
    # Run CCA on already-preprocessed data (bypass preprocess step)
    pred_labels: list[str] = []
    scores_list: list[list[float]] = []
    for epoch in X_proc:
        lbl, sc = pipeline.classifier.predict_label(epoch, labels)
        pred_labels.append(lbl)
        scores_list.append(sc)

    n_correct = sum(p == t for p, t in zip(pred_labels, true_labels_str))
    results = {
        "accuracy": n_correct / n_trials,
        "n_correct": n_correct,
        "n_total": n_trials,
        "predictions": pred_labels,
        "cca_scores": scores_list,
    }

    # Print summary
    print()
    print("=" * 60)
    print(f"  Accuracy : {results['accuracy']:.1%}  ({results['n_correct']}/{results['n_total']})")
    print("=" * 60)
    header = f"{'Trial':>6}  {'True':>6}  {'Pred':>6}  " + "  ".join(f"CCA_{l:>5}" for l in labels) + "  OK"
    print(header)
    print("-" * len(header))
    for i, (true, pred, sc) in enumerate(zip(true_labels_str, pred_labels, scores_list)):
        ok = "v" if true == pred else "x"
        score_str = "  ".join(f"{s:.4f}" for s in sc)
        print(f"{i+1:>6}  {true:>6}  {pred:>6}  {score_str}  {ok}")

    if not args.no_plot:
        visualize(
            X_raw, X_proc, y, results,
            frequencies_hz=frequencies_hz,
            sample_rate_hz=args.sample_rate,
            labels=labels,
            save_dir=processed_dir,
        )


if __name__ == "__main__":
    main()
