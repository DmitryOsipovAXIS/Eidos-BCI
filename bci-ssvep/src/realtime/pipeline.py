"""SSVEP collection with a start menu and single/alternating modes."""
from __future__ import annotations
from utils.config import SSVEPConfig
from ui.screens import one_box_live, run_alternating, run_live, run_single, start_menu
from pipeline.realtime_pipeline import RealtimeCCAPipeline
from acquisition.recording import init_board

import argparse
import asyncio
import dataclasses
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MONITOR_HZ = 60.0
DEFAULT_TARGET_HZ = 7.5
DEFAULT_DURATION_S = 120.0
DEFAULT_DISCARD_S = 1.5

DEFAULT_LEFT_HZ = 7.5
DEFAULT_RIGHT_HZ = 12.0
DEFAULT_ALT_TOTAL_S = 2 * 60.0
DEFAULT_ALT_BLOCK_S = 30.0


@dataclasses.dataclass
class LiveCollectArgs:
    serial_port: str | None = None
    num_channels: int = 8
    no_eeg: bool = False
    fullscreen: bool = False
    monitor_hz: float = DEFAULT_MONITOR_HZ
    discard: float = DEFAULT_DISCARD_S
    mode: str = "menu"
    single_hz: float = DEFAULT_TARGET_HZ
    single_side: str = "left"
    single_duration: float = DEFAULT_DURATION_S
    left_hz: float = DEFAULT_LEFT_HZ
    right_hz: float = DEFAULT_RIGHT_HZ
    alt_total: float = DEFAULT_ALT_TOTAL_S
    alt_block: float = DEFAULT_ALT_BLOCK_S
    alt_start: int = 0
    both_flicker: bool = False
    save: bool = False
    window_s: float = 2.0
    step_s: float = 1.0
    confidence_ratio: float = 1.3
    min_score: float = 0.02


def _sel(selection: Optional[dict], key: str, fallback):
    """Return selection[key] if selection is set, otherwise fallback."""
    return selection.get(key, fallback) if selection else fallback


def _teardown_stream(stream) -> None:
    try:
        stream.stop_stream()
    except Exception:
        pass
    stream.release_session()


def _broadcast_no_eeg(broadcast, loop) -> None:
    if broadcast is not None and loop is not None:
        asyncio.run_coroutine_threadsafe(broadcast("No EEG recorded."), loop)
    print("No EEG recorded.", flush=True)


def _start_rt_pipeline(
    stream,
    frequencies_hz: list[float],
    fs: float,
    args: LiveCollectArgs,
) -> RealtimeCCAPipeline:
    rt_pipeline = RealtimeCCAPipeline(
        stream=stream,
        frequencies_hz=frequencies_hz,
        sample_rate_hz=fs,
        window_s=args.window_s,
        step_s=args.step_s,
        confidence_ratio=args.confidence_ratio,
        min_absolute=args.min_score,
    )
    rt_pipeline.start()
    print(
        f"Real-time CCA pipeline started (window={args.window_s}s, step={args.step_s}s)", flush=True)
    return rt_pipeline


def _compute_psd(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(2048, eeg.shape[1])
    psds = []
    freqs = None
    for ch in range(eeg.shape[0]):
        freqs, pxx = welch(eeg[ch], fs=fs, nperseg=nperseg)
        psds.append(pxx)
    mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
    return freqs, mean_psd


def _print_psd_summary(eeg: np.ndarray, fs: float, target_hz: float) -> None:
    freqs, mean_psd = _compute_psd(eeg, fs)
    mask = (freqs >= target_hz - 2.0) & (freqs <= target_hz + 2.0)
    peak_freq = float(freqs[mask][np.argmax(mean_psd[mask])]) if np.any(
        mask) else float(freqs[np.argmax(mean_psd)])
    peak_power = float(np.max(mean_psd[mask])) if np.any(
        mask) else float(np.max(mean_psd))
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    print(
        f"Recorded PSD shape={mean_psd.shape} (mean over channels)", flush=True)
    print(
        f"Peak freq in target band={peak_freq:0.3f} Hz (power={peak_power:0.3g})", flush=True)
    print(
        f"Power at {target_hz:0.3f} Hz -> bin {freqs[idx]:0.3f} Hz, power={mean_psd[idx]:0.3g}", flush=True)


def _show_report(
    eeg: Optional[np.ndarray],
    fs: float,
    frequencies_hz: list[float],
    all_guesses: list[str],
) -> None:
    from collections import Counter
    from pipeline.preprocessing import common_average_reference
    from pipeline.signal_filtering import apply_filters

    counts = Counter(all_guesses)
    total = len(all_guesses)
    print("\n" + "=" * 40, flush=True)
    print("  LIVE CLASSIFICATION REPORT", flush=True)
    print("=" * 40, flush=True)
    print(f"  Total guesses : {total}", flush=True)
    for lbl in ["LEFT", "RIGHT"]:
        n = counts.get(lbl, 0)
        pct = (n / total * 100) if total > 0 else 0.0
        print(f"  {lbl:>6}        : {n}  ({pct:.1f}%)", flush=True)
    print("=" * 40 + "\n", flush=True)

    if eeg is None or eeg.shape[1] < 10:
        return

    proc = common_average_reference(eeg)
    proc = apply_filters(proc, fs)
    n_ch, n_samples = eeg.shape
    t = np.arange(n_samples) / fs
    freqs_fft = np.fft.rfftfreq(n_samples, 1.0 / fs)

    fig, axes = plt.subplots(n_ch, 2, figsize=(
        14, max(6, n_ch * 1.8)), sharex="col")
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    for ch in range(n_ch):
        axes[ch, 0].plot(t, eeg[ch], lw=0.7, alpha=0.8)
        axes[ch, 1].plot(t, proc[ch], lw=0.7, alpha=0.8, color="orange")
        axes[ch, 0].set_ylabel(f"CH{ch}")
        axes[ch, 1].set_ylabel(f"CH{ch}")
    axes[0, 0].set_title("Raw EEG")
    axes[0, 1].set_title("Filtered (CAR + notch + bandpass)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Raw vs Filtered EEG", fontsize=13)
    fig.tight_layout()

    fft_mag = np.abs(np.fft.rfft(proc, axis=-1)) ** 2
    mean_psd = fft_mag.mean(axis=0)
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs_fft, mean_psd, lw=1.2)
    for f in frequencies_hz:
        ax.axvline(f, color="red", lw=1.2, ls="--", alpha=0.8, label=f"{f} Hz")
        for h in range(2, 4):
            ax.axvline(f * h, color="orange", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("PSD — mean over channels (filtered)")
    ax.legend()
    fig2.tight_layout()

    plt.show()


def _save_alt_results(
    args: LiveCollectArgs,
    eeg_list: list[np.ndarray],
    left_hz: float,
    right_hz: float,
) -> None:
    config = SSVEPConfig(left_hz=left_hz, right_hz=right_hz)
    config.data_raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    min_len = min(x.shape[1] for x in eeg_list)
    X = np.stack([x[:, :min_len] for x in eeg_list])
    y = np.array([(args.alt_start + i) %
                 2 for i in range(len(eeg_list))], dtype=np.int64)
    x_path = config.data_raw_dir / f"X_{ts}.npy"
    y_path = config.data_raw_dir / f"y_{ts}.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved raw data -> {x_path}  {y_path}", flush=True)


def _save_single_result(
    args: LiveCollectArgs,
    eeg: np.ndarray,
    hz: float,
    label: int,
) -> None:
    config = SSVEPConfig(left_hz=hz, right_hz=hz)
    config.data_raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    X = np.expand_dims(eeg, axis=0)
    y = np.array([label], dtype=np.int64)
    x_path = config.data_raw_dir / f"X_{ts}.npy"
    y_path = config.data_raw_dir / f"y_{ts}.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved raw data -> {x_path}  {y_path}", flush=True)


def _run_alt_mode(
    args: LiveCollectArgs,
    stream,
    fs: Optional[float],
    left_hz: float,
    right_hz: float,
    alt_total_s: float,
    alt_block_s: float,
    both_flicker: bool,
    frequencies_hz: list[float],
    rt_pipeline: Optional[RealtimeCCAPipeline],
    broadcast,
    loop,
) -> None:
    eeg_list = run_alternating(
        args, stream, fs,
        left_hz, right_hz, alt_total_s, alt_block_s,
        args.alt_start, rt_pipeline, both_flicker,
    )

    if rt_pipeline is not None:
        rt_pipeline.stop()
    if stream is not None:
        _teardown_stream(stream)

    all_guesses = rt_pipeline.get_full_history() if rt_pipeline is not None else []

    if not eeg_list:
        _broadcast_no_eeg(broadcast, loop)
        _show_report(None, fs or 125.0, frequencies_hz, all_guesses)
        return

    _save_alt_results(args, eeg_list, left_hz, right_hz)
    _show_report(eeg_list[0], fs or 125.0, frequencies_hz, all_guesses)


def _run_single_mode(
    args: LiveCollectArgs,
    stream,
    fs: Optional[float],
    side: str,
    hz: float,
    duration_s: float,
    label: int,
    frequencies_hz: list[float],
    rt_pipeline: Optional[RealtimeCCAPipeline],
    broadcast,
    loop,
) -> None:
    eeg = run_single(args, stream, fs, side, hz,
                     duration_s, label, rt_pipeline)

    if rt_pipeline is not None:
        rt_pipeline.stop()
    if stream is not None:
        _teardown_stream(stream)

    if eeg is None:
        _broadcast_no_eeg(broadcast, loop)
        return

    print(f"Raw EEG shape={eeg.shape}", flush=True)
    for ch in range(eeg.shape[0]):
        ch_data = eeg[ch]
        print(
            f"  CH{ch}: min={float(ch_data.min()):0.6g}, max={float(ch_data.max()):0.6g}, "
            f"mean={float(ch_data.mean()):0.6g}, std={float(ch_data.std()):0.6g}",
            flush=True,
        )
    _print_psd_summary(eeg, fs or 125.0, hz)

    all_guesses = rt_pipeline.get_full_history() if rt_pipeline is not None else []
    _save_single_result(args, eeg, hz, label)
    _show_report(eeg, fs or 125.0, frequencies_hz, all_guesses)


def start_live(args: LiveCollectArgs | None = None, broadcast=None, loop=None) -> None:
    if args is None:
        args = LiveCollectArgs()

    stream = None
    fs: Optional[float] = None

    if args.mode == "menu" and not args.no_eeg:
        print("Initializing board...", flush=True)
        stream, fs = init_board(args)

    selection = None
    if args.mode == "menu":
        selection = start_menu(args.fullscreen)
        if selection.get("mode") != "cli":
            args.mode = selection["mode"]

    if args.mode == "single":
        side = _sel(selection, "side", args.single_side)
        hz = _sel(selection, "hz", args.single_hz)
        duration_s = _sel(selection, "duration", args.single_duration)
        label = 0 if side == "left" else 1
        frequencies_hz = [args.left_hz, args.right_hz]
        print("collecten: single-target test", flush=True)
        print(
            f"duration={duration_s:0.1f}s, target={hz:0.2f} Hz, label={label}", flush=True)
    elif args.mode == "alt":
        left_hz = _sel(selection, "left_hz", args.left_hz)
        right_hz = _sel(selection, "right_hz", args.right_hz)
        alt_total_s = _sel(selection, "alt_total", args.alt_total)
        alt_block_s = _sel(selection, "alt_block", args.alt_block)
        both_flicker = _sel(selection, "both_flicker", args.both_flicker)
        frequencies_hz = [left_hz, right_hz]
        print("collecten: alternating test", flush=True)
        print(f"total={alt_total_s:0.1f}s, block={alt_block_s:0.1f}s, left={left_hz:0.2f} Hz, right={right_hz:0.2f} Hz", flush=True)
    elif args.mode == "live":
        frequencies_hz = [args.left_hz, args.right_hz]
        print(f"live mode: left={args.left_hz:.2f} Hz, right={args.right_hz:.2f} Hz", flush=True)
    elif args.mode == "one_box_live":
        frequencies_hz = [args.right_hz]
        print(f"one-box live mode: {args.right_hz:.2f} Hz", flush=True)
    else:
        print("collecten: using CLI settings", flush=True)
        side = args.single_side
        hz = args.single_hz
        duration_s = args.single_duration
        label = 0 if side == "left" else 1
        frequencies_hz = [args.left_hz, args.right_hz]
        both_flicker = args.both_flicker

    if stream is None and not args.no_eeg:
        stream, fs = init_board(args)

    rt_pipeline: Optional[RealtimeCCAPipeline] = None
    if stream is not None:
        rt_pipeline = _start_rt_pipeline(
            stream, frequencies_hz, fs or 125.0, args)
    if args.mode == "live":
        run_live(args, args.left_hz, args.right_hz, rt_pipeline)
    elif args.mode == "one_box_live":
        one_box_live(args, args.right_hz, rt_pipeline)
    elif args.mode == "alt":
        _run_alt_mode(
            args, stream, fs,
            left_hz, right_hz, alt_total_s, alt_block_s, both_flicker,
            frequencies_hz, rt_pipeline, broadcast, loop,
        )
    else:
        _run_single_mode(
            args, stream, fs,
            side, hz, duration_s, label,
            frequencies_hz, rt_pipeline, broadcast, loop,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSVEP collection with start menu")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--monitor-hz", type=float, default=DEFAULT_MONITOR_HZ)
    parser.add_argument("--discard", type=float, default=DEFAULT_DISCARD_S)
    parser.add_argument(
        "--mode", choices=["menu", "single", "alt"], default="menu")
    parser.add_argument("--single-hz", type=float, default=DEFAULT_TARGET_HZ)
    parser.add_argument(
        "--single-side", choices=["left", "right"], default="left")
    parser.add_argument("--single-duration", type=float,
                        default=DEFAULT_DURATION_S)
    parser.add_argument("--left-hz", type=float, default=DEFAULT_LEFT_HZ)
    parser.add_argument("--right-hz", type=float, default=DEFAULT_RIGHT_HZ)
    parser.add_argument("--alt-total", type=float, default=DEFAULT_ALT_TOTAL_S)
    parser.add_argument("--alt-block", type=float, default=DEFAULT_ALT_BLOCK_S)
    parser.add_argument("--alt-start", type=int, choices=[0, 1], default=0)
    parser.add_argument("--both-flicker", action="store_true",
                        help="During alternating blocks, flicker both targets while cueing focus")
    parser.add_argument("--save", action="store_true",
                        help="Save raw X/y to data/raw")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--confidence-ratio", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.02)
    parsed = parser.parse_args()

    start_live(LiveCollectArgs(
        serial_port=parsed.serial_port,
        num_channels=parsed.num_channels,
        no_eeg=parsed.no_eeg,
        fullscreen=parsed.fullscreen,
        monitor_hz=parsed.monitor_hz,
        discard=parsed.discard,
        mode=parsed.mode,
        single_hz=parsed.single_hz,
        single_side=parsed.single_side,
        single_duration=parsed.single_duration,
        left_hz=parsed.left_hz,
        right_hz=parsed.right_hz,
        alt_total=parsed.alt_total,
        alt_block=parsed.alt_block,
        alt_start=parsed.alt_start,
        both_flicker=parsed.both_flicker,
        save=parsed.save,
        window_s=parsed.window_s,
        step_s=parsed.step_s,
        confidence_ratio=parsed.confidence_ratio,
        min_score=parsed.min_score,
    ))
