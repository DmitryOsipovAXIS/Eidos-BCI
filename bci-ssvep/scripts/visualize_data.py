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
from scipy.signal import welch, spectrogram

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


def mean_psd_for_trials(trials: np.ndarray, fs: float, low: float, high: float, n_samples: int, n_ch: int):
    """Return frequency bins and mean/std PSD for a set of trials.

    If no trials are present, returns ``(None, None, None)``.
    """
    if trials.size == 0:
        return None, None, None

    psds = []
    freqs = None
    for trial in trials:
        filtered = bandpass_filter(trial, fs, low, high)
        ch_psds = []
        for ch in range(n_ch):
            freqs, pxx = welch(filtered[ch], fs=fs, nperseg=min(256, n_samples))
            ch_psds.append(pxx)
        psds.append(np.mean(ch_psds, axis=0))

    psds = np.asarray(psds)
    return freqs, np.mean(psds, axis=0), np.std(psds, axis=0)


def time_frequency_view(trial: np.ndarray, fs: float, low: float, high: float):
    """Return a channel-averaged spectrogram and a dominant-frequency trace."""
    filtered = bandpass_filter(trial, fs, low, high)
    nperseg = int(max(64, min(fs * 4.0, filtered.shape[1])))
    noverlap = int(nperseg * 0.75)

    specs = []
    freqs = None
    times = None
    for ch in range(filtered.shape[0]):
        freqs, times, sxx = spectrogram(
            filtered[ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )
        specs.append(sxx)

    mean_spec = np.mean(np.stack(specs, axis=0), axis=0)
    band_mask = (freqs >= low) & (freqs <= high)
    if np.any(band_mask):
        band_freqs = freqs[band_mask]
        band_spec = mean_spec[band_mask]
        dominant_idx = np.argmax(band_spec, axis=0)
        dominant_freq = band_freqs[dominant_idx]
    else:
        dominant_freq = np.full(times.shape, np.nan)

    return freqs, times, mean_spec, dominant_freq


def selected_channel_indices(n_ch: int) -> list[int]:
    """Prefer the occipital channels we care about most."""
    return list(range(min(4, n_ch)))


def selected_channel_names(n_ch: int) -> list[str]:
    base = ["POz", "O1", "Oz", "O2"]
    return base[:min(4, n_ch)]


def peak_frequency_over_time(trial: np.ndarray, fs: float, low: float, high: float, window_s: float = 1.0):
    """Return the strongest frequency in the band for each non-overlapping time window."""
    filtered = bandpass_filter(trial, fs, low, high)
    window_samples = int(max(1, round(window_s * fs)))
    n_windows = filtered.shape[1] // window_samples
    if n_windows == 0:
        return np.array([]), np.array([])

    times = []
    peaks = []
    for wi in range(n_windows):
        start = wi * window_samples
        end = start + window_samples
        chunk = filtered[:, start:end]
        ch_psds = []
        freqs = None
        for ch in range(chunk.shape[0]):
            freqs, pxx = welch(chunk[ch], fs=fs, nperseg=min(256, chunk.shape[1]))
            ch_psds.append(pxx)
        mean_psd = np.mean(np.stack(ch_psds, axis=0), axis=0)
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            local_freqs = freqs[band_mask]
            local_psd = mean_psd[band_mask]
            peak_freq = float(local_freqs[np.argmax(local_psd)])
        else:
            peak_freq = float("nan")
        peaks.append(peak_freq)
        times.append((start + end) / 2.0 / fs)

    return np.asarray(times), np.asarray(peaks)


def sliding_window_peak_frequency_over_time(
    trial: np.ndarray,
    fs: float,
    low: float,
    high: float,
    window_s: float = 4.0,
    step_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest frequency in the band for overlapping sliding windows."""
    filtered = bandpass_filter(trial, fs, low, high)
    window_samples = int(max(1, round(window_s * fs)))
    step_samples = int(max(1, round(step_s * fs)))
    if filtered.shape[1] < window_samples:
        return np.array([]), np.array([])

    times = []
    peaks = []
    for start in range(0, filtered.shape[1] - window_samples + 1, step_samples):
        end = start + window_samples
        chunk = filtered[:, start:end]
        ch_psds = []
        freqs = None
        for ch in range(chunk.shape[0]):
            freqs, pxx = welch(chunk[ch], fs=fs, nperseg=min(512, chunk.shape[1]))
            ch_psds.append(pxx)
        mean_psd = np.mean(np.stack(ch_psds, axis=0), axis=0)
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            local_freqs = freqs[band_mask]
            local_psd = mean_psd[band_mask]
            peak_freq = float(local_freqs[np.argmax(local_psd)])
        else:
            peak_freq = float("nan")
        peaks.append(peak_freq)
        times.append((start + end) / 2.0 / fs)

    return np.asarray(times), np.asarray(peaks)


def per_channel_peak_frequency_over_time(trial: np.ndarray, fs: float, low: float, high: float, window_s: float = 1.0):
    """Return a peak-frequency trace for each channel using 1-second windows."""
    filtered = bandpass_filter(trial, fs, low, high)
    window_samples = int(max(1, round(window_s * fs)))
    n_windows = filtered.shape[1] // window_samples
    if n_windows == 0:
        return np.array([]), np.array([[]])

    times = []
    traces = []
    for wi in range(n_windows):
        start = wi * window_samples
        end = start + window_samples
        chunk = filtered[:, start:end]
        channel_peaks = []
        for ch in range(chunk.shape[0]):
            freqs_ref, pxx = welch(chunk[ch], fs=fs, nperseg=min(256, chunk.shape[1]))
            band_mask = (freqs_ref >= low) & (freqs_ref <= high)
            if np.any(band_mask):
                local_freqs = freqs_ref[band_mask]
                local_psd = pxx[band_mask]
                channel_peaks.append(float(local_freqs[np.argmax(local_psd)]))
            else:
                channel_peaks.append(float("nan"))
        traces.append(channel_peaks)
        times.append((start + end) / 2.0 / fs)

    return np.asarray(times), np.asarray(traces).T


def band_power_and_snr_over_time(
    trial: np.ndarray,
    fs: float,
    target_hz: float,
    low: float,
    high: float,
    channel_indices: list[int],
    window_s: float = 4.0,
    step_s: float = 1.0,
    guard_hz: float = 0.4,
    noise_band_hz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target-band power and a simple SNR estimate over time."""
    filtered = bandpass_filter(trial, fs, low, high)
    window_samples = int(max(1, round(window_s * fs)))
    step_samples = int(max(1, round(step_s * fs)))
    if filtered.shape[1] < window_samples:
        return np.array([]), np.array([]), np.array([])

    times = []
    target_powers = []
    snr_db = []

    for start in range(0, filtered.shape[1] - window_samples + 1, step_samples):
        end = start + window_samples
        chunk = filtered[channel_indices, start:end]
        ch_psds = []
        freqs = None
        for ch in range(chunk.shape[0]):
            freqs, pxx = welch(chunk[ch], fs=fs, nperseg=min(512, chunk.shape[1]))
            ch_psds.append(pxx)
        mean_psd = np.mean(np.stack(ch_psds, axis=0), axis=0)

        target_idx = int(np.argmin(np.abs(freqs - target_hz)))
        target_power = float(mean_psd[target_idx])

        noise_mask = (
            ((freqs >= target_hz - noise_band_hz) & (freqs <= target_hz - guard_hz)) |
            ((freqs >= target_hz + guard_hz) & (freqs <= target_hz + noise_band_hz))
        ) & (freqs >= low) & (freqs <= high)
        noise_power = float(np.mean(mean_psd[noise_mask])) if np.any(noise_mask) else float("nan")
        snr = 10.0 * np.log10(target_power / noise_power) if np.isfinite(noise_power) and noise_power > 0 else float("nan")

        times.append((start + end) / 2.0 / fs)
        target_powers.append(target_power)
        snr_db.append(snr)

    return np.asarray(times), np.asarray(target_powers), np.asarray(snr_db)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Path to X_*.npy file")
    args = parser.parse_args()

    config = SSVEPConfig(left_hz=7.5, right_hz=15.0)
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
    occipital_idx = selected_channel_indices(n_ch)
    occipital_names = selected_channel_names(n_ch)

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
        f, mean_psd, std_psd = mean_psd_for_trials(trials, fs, low, high, n_samples, n_ch)
        if f is None:
            ax.text(0.5, 0.5, f"No trials for {label}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_xlim(2, 20)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (log scale)")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            continue

        mean_psd = np.asarray(mean_psd)
        std_psd = np.asarray(std_psd)
        safe_lower = np.clip(mean_psd - std_psd, 1e-12, None)
        safe_upper = np.clip(mean_psd + std_psd, 1e-12, None)

        ax.semilogy(f, mean_psd, color=color, linewidth=2, label=label)
        ax.fill_between(f, safe_lower, safe_upper, alpha=0.2, color=color)
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
    fig2, axes2 = plt.subplots(len(occipital_idx), 2, figsize=(14, max(6, len(occipital_idx) * 1.8)), sharex=True)
    fig2.suptitle("Raw EEG — example LEFT trial (left) vs RIGHT trial (right)", fontsize=13)

    ch_names = occipital_names

    example_trials = [left_trials[0] if len(left_trials) else None,
                      right_trials[0] if len(right_trials) else None]

    for plot_row, ci in enumerate(occipital_idx):
        for col, (trial, color, side_label) in enumerate(zip(
            example_trials,
            ["steelblue", "tomato"],
            ["LEFT", "RIGHT"],
        )):
            ax = axes2[plot_row, col]
            if trial is None:
                ax.text(0.5, 0.5, f"No {side_label} trials", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_axis_off()
                continue
            ax.plot(t, trial[ci], color=color, linewidth=0.6)
            ax.set_ylabel(ch_names[plot_row] if plot_row < len(ch_names) else f"Ch{ci}",
                          fontsize=8)
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)
            if plot_row == 0:
                ax.set_title(f"{side_label} trial")
            if plot_row == len(occipital_idx) - 1:
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
    if np.any(y == 0):
        ax3.scatter(feats[y == 0, 0], feats[y == 0, 1],
                    color="steelblue", label="LEFT",  alpha=0.7, s=60)
    if np.any(y == 1):
        ax3.scatter(feats[y == 1, 0], feats[y == 1, 1],
                    color="tomato",    label="RIGHT", alpha=0.7, s=60)
    ax3.set_xlabel(f"log power @ {config.left_hz} Hz")
    ax3.set_ylabel(f"log power @ {config.right_hz} Hz")
    ax3.set_title("Feature space — are LEFT and RIGHT separable?")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 4: time-frequency view — where the dominant frequency moves
    # ------------------------------------------------------------------
    if len(left_trials):
        ref_trial = left_trials[0]
        ref_label = f"LEFT trial @ {config.left_hz} Hz"
    elif len(right_trials):
        ref_trial = right_trials[0]
        ref_label = f"RIGHT trial @ {config.right_hz} Hz"
    else:
        ref_trial = None
        ref_label = "No trial available"

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig4.suptitle("Time-frequency view — band-limited dominant Hertz over time", fontsize=13)

    if ref_trial is None:
        ax4a.text(0.5, 0.5, "No trial available", ha="center", va="center", transform=ax4a.transAxes)
        ax4b.axis("off")
    else:
        freqs_t, times_t, spec_t, dom_freq = time_frequency_view(ref_trial[occipital_idx], fs, low, high)
        band_mask = (freqs_t >= 2.0) & (freqs_t <= 20.0)
        spec_db = 10.0 * np.log10(np.clip(spec_t[band_mask], 1e-12, None))
        im = ax4a.pcolormesh(times_t, freqs_t[band_mask], spec_db, shading="auto", cmap="magma")
        ax4a.axhline(config.left_hz, color="steelblue", linestyle="--", alpha=0.8, label=f"{config.left_hz} Hz target")
        ax4a.axhline(config.right_hz, color="tomato", linestyle="--", alpha=0.8, label=f"{config.right_hz} Hz target")
        ax4a.set_ylabel("Frequency (Hz)")
        ax4a.set_title(ref_label)
        ax4a.legend(loc="upper right")
        cbar = fig4.colorbar(im, ax=ax4a, pad=0.01)
        cbar.set_label("Power (dB)")

        ax4b.plot(times_t, dom_freq, color="black", linewidth=1.5)
        ax4b.axhline(config.left_hz, color="steelblue", linestyle="--", alpha=0.8)
        ax4b.axhline(config.right_hz, color="tomato", linestyle="--", alpha=0.8)
        ax4b.set_ylim(max(2.0, low), min(20.0, high))
        ax4b.set_xlabel("Time (s)")
        ax4b.set_ylabel("Dominant Hz")
        ax4b.grid(True, alpha=0.3)
        ax4b.set_title("Dominant frequency over time (4 s sliding window)")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 5: peak frequency per 1-second window
    # ------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(14, 4))
    fig5.suptitle("Peak frequency per 1-second window", fontsize=13)
    if ref_trial is None:
        ax5.text(0.5, 0.5, "No trial available", ha="center", va="center", transform=ax5.transAxes)
    else:
        sec_times, sec_peaks = peak_frequency_over_time(ref_trial, fs, low, high, window_s=1.0)
        ax5.step(sec_times, sec_peaks, where="mid", color="black", linewidth=1.8, label="1 s peak frequency")
        ax5.axhline(config.left_hz, color="steelblue", linestyle="--", alpha=0.8, label=f"{config.left_hz} Hz target")
        ax5.axhline(config.right_hz, color="tomato", linestyle="--", alpha=0.8, label=f"{config.right_hz} Hz target")
        ax5.set_ylim(max(2.0, low), min(20.0, high))
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Peak Hz")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc="upper right")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 6: per-channel peak frequency traces (1-second windows)
    # ------------------------------------------------------------------
    fig6, axes6 = plt.subplots(len(occipital_idx), 1, figsize=(14, max(6, len(occipital_idx) * 1.4)), sharex=True)
    fig6.suptitle("Per-channel peak frequency over time (1-second windows)", fontsize=13)
    if len(occipital_idx) == 1:
        axes6 = [axes6]

    if ref_trial is None:
        axes6[0].text(0.5, 0.5, "No trial available", ha="center", va="center", transform=axes6[0].transAxes)
        axes6[0].set_axis_off()
    else:
        ch_times, ch_traces = per_channel_peak_frequency_over_time(ref_trial[occipital_idx], fs, low, high, window_s=1.0)
        for ci, ax in enumerate(axes6):
            ax.step(ch_times, ch_traces[ci], where="mid", color="black", linewidth=1.2)
            ax.axhline(config.left_hz, color="steelblue", linestyle="--", alpha=0.8)
            ax.axhline(config.right_hz, color="tomato", linestyle="--", alpha=0.8)
            ax.set_ylim(max(2.0, low), min(20.0, high))
            ax.set_ylabel(occipital_names[ci], fontsize=8)
            ax.grid(True, alpha=0.25)
            if ci == 0:
                ax.set_title("Channel peak frequency")
            if ci == len(occipital_idx) - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 7: target power and simple SNR over time
    # ------------------------------------------------------------------
    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig7.suptitle("Target power over time on occipital channels", fontsize=13)
    if ref_trial is None:
        ax7a.text(0.5, 0.5, "No trial available", ha="center", va="center", transform=ax7a.transAxes)
        ax7b.axis("off")
    else:
        sec_times_75, power_75, snr_75 = band_power_and_snr_over_time(ref_trial, fs, config.left_hz, low, high, occipital_idx, window_s=4.0, step_s=1.0)
        sec_times_15, power_15, snr_15 = band_power_and_snr_over_time(ref_trial, fs, config.right_hz, low, high, occipital_idx, window_s=4.0, step_s=1.0)
        ax7a.plot(sec_times_75, power_75, color="steelblue", linewidth=1.7, label=f"{config.left_hz} Hz power")
        ax7a.plot(sec_times_15, power_15, color="tomato", linewidth=1.7, label=f"{config.right_hz} Hz power")
        ax7a.set_ylabel("Power")
        ax7a.set_yscale("log")
        ax7a.grid(True, alpha=0.25)
        ax7a.legend(loc="upper right")
        ax7a.set_title("Target-band power")

        ax7b.plot(sec_times_75, snr_75, color="steelblue", linewidth=1.7, label=f"{config.left_hz} Hz SNR (dB)")
        ax7b.plot(sec_times_15, snr_15, color="tomato", linewidth=1.7, label=f"{config.right_hz} Hz SNR (dB)")
        ax7b.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax7b.set_ylabel("SNR (dB)")
        ax7b.set_xlabel("Time (s)")
        ax7b.grid(True, alpha=0.25)
        ax7b.legend(loc="upper right")
        ax7b.set_title("Target-versus-nearby-bin SNR")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
