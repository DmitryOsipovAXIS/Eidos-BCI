"""Single-target SSVEP: PsychoPy flicker + live NCCA + spectrum plot (BrainFlow, SciPy).

One full-contrast flashing target whose rate is ``--flicker-hz`` (good choices on a 60 Hz
monitor: e.g. 10, 12, 15, 20 Hz). A Matplotlib window shows the average-channel Welch PSD
(last analysis window); a dashed line marks the stimulus frequency so you can see visually
whether occipital power lines up.

Real-time EEG: causal ``sosfilt`` bandpass 0.5–32 Hz, 50 Hz notch, NCCA (Kartsch et al.,
Sensors 2022, doi:10.3390/s22249803).

Usage (from ``bci-ssvep``):

  python scripts/ssvep_psychopy_live.py --serial-port COM4
  python scripts/ssvep_psychopy_live.py --flicker-hz 10 --fullscreen
  python scripts/ssvep_psychopy_live.py --no-eeg

Quit stimulus: Escape. Close spectrum window or Ctrl+C stops the spectrum thread when you exit.

"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")  # separate spectrum window alongside PsychoPy
import matplotlib.pyplot as plt
import numpy as np
from psychopy import core, event, visual
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, welch as scipy_welch
from sklearn.cross_decomposition import CCA

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from acquisition.brainflow_stream import BrainFlowStream

ANALYSIS_INTERVAL_S = 0.35
WINDOW_SECONDS = 4.0
MONITOR_HZ_FALLBACK = 60

NCCA_DELTA_HZ = 0.2
NCCA_QUALITY_THRESHOLD = 1.2
N_HARMONICS_DEFAULT = 2

# PSD plot band (Hz) for “where is the peak?” hint on HUD
PEAK_SEARCH_HZ = (5.0, 29.0)


# =============================================================================
# Filtering (causal, real-time friendly): bandpass + line notch
# =============================================================================


def design_filter_sos(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Design SOS for 50 Hz notch + 0.5–32 Hz bandpass (causal ``sosfilt`` chain)."""
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    nyq = fs / 2.0

    b_n, a_n = iirnotch(w0=50.0, Q=35.0, fs=fs)
    sos_notch = tf2sos(b_n, a_n)

    lo = max(0.5 / nyq, 1e-5)
    hi = min(32.0 / nyq, 1.0 - 1e-5)
    if hi <= lo:
        raise ValueError("Invalid Nyquist relative to bandpass.")

    sos_band = butter(N=4, Wn=[lo, hi], btype="band", output="sos")
    return sos_notch, sos_band


def filter_eeg_realtime(eeg: np.ndarray, sos_notch: np.ndarray, sos_band: np.ndarray) -> np.ndarray:
    """Notch then bandpass along time (axis=1)."""
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2-D (channels x samples).")
    x = sosfilt(sos_notch, eeg, axis=-1)
    x = sosfilt(sos_band, x, axis=-1)
    return x


# =============================================================================
# Reference + CCA + NCCA
# =============================================================================


def make_reference(freq: float, sfreq: float, n_samples: int, n_harmonics: int = 2) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    cols = []
    for h in range(1, n_harmonics + 1):
        ph = 2.0 * np.pi * freq * h * t
        cols.append(np.sin(ph))
        cols.append(np.cos(ph))
    return np.stack(cols, axis=1)


def compute_cca(eeg: np.ndarray, freq: float, sfreq: float, n_harmonics: int = N_HARMONICS_DEFAULT) -> float:
    _, n_s = eeg.shape
    if n_s < 2 * (n_harmonics + eeg.shape[0]):
        return 0.0

    y_ref = make_reference(freq, sfreq, n_s, n_harmonics)
    x_eeg = eeg.T.copy()
    x_eeg -= x_eeg.mean(axis=0, keepdims=True)
    y_ref = y_ref - y_ref.mean(axis=0, keepdims=True)

    if x_eeg.shape[1] < 1 or y_ref.shape[1] < 1:
        return 0.0

    cca = CCA(n_components=1, max_iter=800)
    try:
        cca.fit(x_eeg, y_ref)
        x_c, y_c = cca.transform(x_eeg, y_ref)
        r = np.corrcoef(x_c.ravel(), y_c.ravel())[0, 1]
        if not np.isfinite(r):
            return 0.0
        return float(np.clip(abs(r), 0.0, 1.0))
    except Exception:
        return 0.0


def compute_ncca(eeg: np.ndarray, freq: float, sfreq: float, delta: float = NCCA_DELTA_HZ) -> float:
    c0 = compute_cca(eeg, freq, sfreq)
    cp = compute_cca(eeg, freq + delta, sfreq)
    cm = compute_cca(eeg, freq - delta, sfreq)
    denom = 0.5 * (cp + cm)
    if denom < 1e-8:
        return float("inf") if c0 > 1e-8 else 0.0
    return float(c0 / denom)


class LiveMetrics:
    """HUD + spectrum plot snapshots (single target frequency)."""

    def __init__(self, target_hz: float) -> None:
        self.lock = threading.Lock()
        self.target_hz = float(target_hz)
        self.score_hud: float = float("nan")
        self.quality_ok: bool = False
        self.peak_hz: float = float("nan")
        self.note: str = ""
        self.spec_f: np.ndarray | None = None
        self.spec_p: np.ndarray | None = None

    def update(
        self,
        *,
        score_hud: float,
        quality_ok: bool,
        peak_hz: float,
        spec_f: np.ndarray,
        spec_p: np.ndarray,
        note: str,
    ) -> None:
        with self.lock:
            self.score_hud = score_hud
            self.quality_ok = quality_ok
            self.peak_hz = peak_hz
            self.note = note[:200]
            self.spec_f = spec_f.astype(np.float64).copy()
            self.spec_p = spec_p.astype(np.float64).copy()

    def label(self) -> str:
        with self.lock:
            f = self.target_hz
            s = (
                float(self.score_hud)
                if np.isfinite(self.score_hud)
                else float("nan")
            )
            sc = "—" if not np.isfinite(s) else f"{s:.2f}"
            ph = (
                "—"
                if not np.isfinite(self.peak_hz)
                else f"{self.peak_hz:.1f}"
            )
            qw = "OK" if self.quality_ok else "LOW"
            line1 = f"NCCA @ {f:.0f} Hz: {sc}  (quality {qw}, thresh {NCCA_QUALITY_THRESHOLD:.1f})"
            line2 = f"Welch peak {PEAK_SEARCH_HZ[0]:.0f}–{PEAK_SEARCH_HZ[1]:.0f} Hz: {ph} Hz  |  dashed line = stim"
            decision = f">>> DETECTED: {f:.0f} Hz <<<" if self.quality_ok else ">>> NO DETECTION <<<"
            return f"{line1}\n{line2}\n{decision}\n{self.note}"

    def spectrum_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self.lock:
            if self.spec_f is None or self.spec_p is None:
                return None, None
            return self.spec_f.copy(), self.spec_p.copy()


def _eeg_loop(
    stream: BrainFlowStream,
    target_hz: float,
    metrics: LiveMetrics,
    stop_evt: threading.Event,
) -> None:
    fs = stream.sampling_rate()
    win_n = max(int(WINDOW_SECONDS * fs), 256)
    sos_notch, sos_band = design_filter_sos(fs)
    time.sleep(WINDOW_SECONDS * 0.2)

    while not stop_evt.wait(ANALYSIS_INTERVAL_S):
        if stream.get_board_data_count() < win_n:
            continue
        raw = stream.get_current_board_data(win_n)
        eeg_ix = stream.eeg_channel_indices()
        eeg = raw[eeg_ix, :].astype(np.float64)

        eeg_f = filter_eeg_realtime(eeg, sos_notch, sos_band)

        v_raw = compute_ncca(eeg_f, target_hz, fs, delta=NCCA_DELTA_HZ)
        if np.isinf(v_raw) and v_raw > 0:
            score_hud = 99.99
        elif not np.isfinite(v_raw):
            score_hud = 0.0
        else:
            score_hud = float(min(v_raw, 99.99))

        if np.isnan(v_raw):
            quality_ok = False
        elif np.isinf(v_raw) and v_raw > 0:
            quality_ok = True
        else:
            quality_ok = v_raw > NCCA_QUALITY_THRESHOLD

        sig_avg = np.mean(eeg_f, axis=0)
        n_seg = int(min(fs * 2, sig_avg.shape[-1]))
        n_seg = max(128, n_seg - (n_seg % 2))
        fq, pw = scipy_welch(sig_avg, fs=fs, nperseg=n_seg, noverlap=n_seg // 2)

        pf_lo, pf_hi = PEAK_SEARCH_HZ
        m = (fq >= pf_lo) & (fq <= pf_hi)
        if np.any(m):
            jj = int(np.argmax(pw[m]))
            peak_hz = float(fq[m][jj])
        else:
            peak_hz = float("nan")

        note = f"SOS 0.5–32 Hz + 50 Hz notch  ·  Welch nperseg={n_seg}  ·  {WINDOW_SECONDS}s window"
        metrics.update(
            score_hud=score_hud,
            quality_ok=quality_ok,
            peak_hz=peak_hz,
            spec_f=fq,
            spec_p=pw,
            note=note,
        )


def _spectrum_plot_loop(target_hz: float, metrics: LiveMetrics, stop_evt: threading.Event) -> None:
    """Refresh a small PSD plot from shared metrics."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#252525")
    ax.tick_params(colors="gray")
    ax.spines[["bottom", "top", "left", "right"]].set_color("gray")
    ax.set_title("Average-channel PSD — look for a bump near the red line", color="silver", fontsize=10)
    fig.canvas.manager.set_window_title("SSVEP spectrum")

    while not stop_evt.is_set():
        time.sleep(max(ANALYSIS_INTERVAL_S, 0.2))
        fq, pw = metrics.spectrum_snapshot()
        if fq is None or pw is None:
            continue

        ax.clear()
        ax.set_facecolor("#252525")
        ax.tick_params(colors="gray")
        for s in ax.spines.values():
            s.set_color("gray")
        ax.semilogy(fq, np.maximum(pw, 1e-20), color="cyan", lw=1.0)
        ax.axvline(target_hz, color="tomato", ls="--", lw=2, label=f"{target_hz:g} Hz (stimulus)")
        ax.set_xlim(2.0, 35.0)
        ax.set_xlabel("Frequency (Hz)", color="silver")
        ax.set_ylabel("PSD", color="silver")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.35)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if not plt.fignum_exists(fig.number):
            break

    plt.close(fig)


def _prepare_knight(serial_port: str, num_channels: int) -> BrainFlowStream:
    BoardShim.disable_board_logger()
    p = BrainFlowInputParams()
    p.serial_port = serial_port
    stream = BrainFlowStream(
        board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
        params=p,
        num_channels=num_channels,
    )
    stream.prepare_session()
    stream.start_stream()
    time.sleep(2.0)

    board = stream.board
    for ch in range(1, 9):
        time.sleep(0.5)
        board.config_board(f"chon_{ch}_12")
        time.sleep(1.0)
        board.config_board(f"rldadd_{ch}")
        time.sleep(0.5)
    print("Board ready.", flush=True)
    return stream


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PsychoPy single-target SSVEP + NCCA + live spectrum.",
    )
    parser.add_argument("--serial-port", default="COM4")
    parser.add_argument(
        "--flicker-hz",
        type=float,
        default=10.0,
        help="Stimulus flicker (Hz); on 60 Hz, 10 / 12 / 15 / 20 work cleanly.",
    )
    parser.add_argument(
        "--monitor-hz",
        type=float,
        default=MONITOR_HZ_FALLBACK,
        help="Fallback dt if frame clock yields non-positive delta.",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true", help="Stimulus without BrainFlow.")

    args = parser.parse_args()
    target_hz = args.flicker_hz

    stop_evt = threading.Event()
    metrics = LiveMetrics(target_hz)
    stream: BrainFlowStream | None = None
    tid_eeg: threading.Thread | None = None
    tid_plot: threading.Thread | None = None

    if not args.no_eeg:
        stream = _prepare_knight(args.serial_port, args.num_channels)
        tid_eeg = threading.Thread(
            target=_eeg_loop,
            kwargs={
                "stream": stream,
                "target_hz": target_hz,
                "metrics": metrics,
                "stop_evt": stop_evt,
            },
            daemon=True,
        )
        tid_eeg.start()

        tid_plot = threading.Thread(
            target=_spectrum_plot_loop,
            args=(target_hz, metrics, stop_evt),
            daemon=True,
        )
        tid_plot.start()

    win = visual.Window(
        fullscr=args.fullscreen,
        color=(-1, -1, -1),
        units="norm",
        waitBlanking=True,
    )

    stimulus = visual.Rect(win, width=0.4, height=0.38, pos=(0.0, 0.08))
    title = visual.TextStim(
        win,
        text="Focus the flashing square — check spectrum window vs red line",
        height=0.055,
        pos=(0.0, 0.82),
        color=(0.85, 0.85, 0.85),
    )
    hz_label = visual.TextStim(
        win,
        text=f"{target_hz:.1f} Hz",
        height=0.075,
        pos=(0.0, -0.26),
        color=(0.35, 0.85, 1.0),
    )
    hint = visual.TextStim(
        win,
        text=f"Nominal display {args.monitor_hz:.0f} Hz  ·  Escape to quit",
        height=0.04,
        pos=(0.0, -0.88),
        color=(0.5, 0.5, 0.5),
    )
    status = visual.TextStim(win, text="", height=0.032, pos=(0.0, -0.50), wrapWidth=1.85)

    if args.no_eeg:
        status.text = "(EEG offline — flicker demo only)"

    clock = core.Clock()
    phase_acc = 0.0
    clock.reset()

    try:
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                break

            dt = clock.getTime()
            clock.reset()
            dt = dt if dt > 0 else 1.0 / args.monitor_hz
            phase_acc = (phase_acc + target_hz * dt) % 1.0

            stimulus.fillColor = (1, 1, 1) if phase_acc < 0.5 else (-1, -1, -1)

            if not args.no_eeg:
                status.text = metrics.label()

            title.draw()
            hz_label.draw()
            stimulus.draw()
            status.draw()
            hint.draw()
            win.flip()
    finally:
        stop_evt.set()
        win.close()
        core.quit()

        if tid_eeg is not None:
            tid_eeg.join(timeout=WINDOW_SECONDS)
        if tid_plot is not None:
            tid_plot.join(timeout=2.0)

        try:
            plt.close("all")
        except Exception:
            pass

        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.release_session()
            except Exception:
                pass


if __name__ == "__main__":
    main()
