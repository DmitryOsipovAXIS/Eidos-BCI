"""Dual-target SSVEP demo: PsychoPy flicker + live NCCA (BrainFlow, SciPy, sklearn).

Two full-contrast flickering targets (default 10 Hz and 15 Hz, divisors of 60 Hz) for
binary SSVEP; occipital montage (e.g. O1, O2, Oz, POz on first four channels, or full
8-channel Knight). Real-time pipeline uses **causal** ``sosfilt`` (SciPy) and
**NCCA** normalized CCA (Kartsch et al., Sensors 2022, doi:10.3390/s22249803).

Dependencies: PsychoPy, BrainFlow (Neuropawn Knight), SciPy, scikit-learn.

Usage (from ``bci-ssvep``):

  python scripts/ssvep_psychopy_live.py --serial-port COM4
  python scripts/ssvep_psychopy_live.py --fullscreen --num-channels 8
  python scripts/ssvep_psychopy_live.py --no-eeg   # stimulus only

``--flicker-hz`` is kept for CLI compatibility; this script uses fixed **10 Hz** and
**15 Hz** targets for classification and on-screen flicker.

Quit: Escape (or Alt+F4 on Windows).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
from psychopy import core, event, visual
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from sklearn.cross_decomposition import CCA

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from acquisition.brainflow_stream import BrainFlowStream

# --- SSVEP targets (incommensurable pair; 15 is not a harmonic of 10) ------------
SSVEP_FREQS: tuple[float, float] = (10.0, 15.0)

ANALYSIS_INTERVAL_S = 0.35
WINDOW_SECONDS = 4.0
MONITOR_HZ_FALLBACK = 60

# NCCA neighbour spacing (Hz) and detection threshold (paper / expert suggestion)
NCCA_DELTA_HZ = 0.2
NCCA_QUALITY_THRESHOLD = 1.2

# Reference signal harmonics for CCA (fundamental + harmonics)
N_HARMONICS_DEFAULT = 2


# =============================================================================
# Filtering (causal, real-time friendly): bandpass + line notch
# =============================================================================


def design_filter_sos(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Design SOS sections for bandpass (0.5–32 Hz) and notch (50 Hz).

    Bandpass limits slow drift and high-frequency noise while keeping SSVEP band.
    European mains: notch at 50 Hz (applied before bandpass so line is attenuated
    even when passband overlaps line harmonics conceptually — here passband ends at
    32 Hz, but notch-first is conventional for EEG preprocessing chains).
    """
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    nyq = fs / 2.0

    # Notch 50 Hz (Q ~ 35: reasonable trade-off between width and stability)
    b_n, a_n = iirnotch(w0=50.0, Q=35.0, fs=fs)
    sos_notch = tf2sos(b_n, a_n)

    lo = max(0.5 / nyq, 1e-5)
    hi = min(32.0 / nyq, 1.0 - 1e-5)
    if hi <= lo:
        raise ValueError("Invalid Nyquist relative to bandpass.")

    sos_band = butter(N=4, Wn=[lo, hi], btype="band", output="sos")
    return sos_notch, sos_band


def filter_eeg_realtime(eeg: np.ndarray, sos_notch: np.ndarray, sos_band: np.ndarray) -> np.ndarray:
    """Apply causal IIR filtering along time (axis=1). EEG shape (n_channels, n_samples)."""
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2-D (channels x samples).")
    # Notch then bandpass; sosfilt is causal (no filtfilt / zero-phase)
    x = sosfilt(sos_notch, eeg, axis=-1)
    x = sosfilt(sos_band, x, axis=-1)
    return x


# =============================================================================
# Reference + CCA + NCCA
# =============================================================================


def make_reference(freq: float, sfreq: float, n_samples: int, n_harmonics: int = 2) -> np.ndarray:
    """Sin/cos reference per harmonic; shape (n_samples, 2 * n_harmonics)."""
    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    cols = []
    for h in range(1, n_harmonics + 1):
        ph = 2.0 * np.pi * freq * h * t
        cols.append(np.sin(ph))
        cols.append(np.cos(ph))
    return np.stack(cols, axis=1)


def compute_cca(eeg: np.ndarray, freq: float, sfreq: float, n_harmonics: int = N_HARMONICS_DEFAULT) -> float:
    """First canonical correlation between multi-channel EEG and sin/cos reference."""
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
    """Normalized CCA: rho(f) / mean(rho(f+delta), rho(f-delta))."""
    c0 = compute_cca(eeg, freq, sfreq)
    cp = compute_cca(eeg, freq + delta, sfreq)
    cm = compute_cca(eeg, freq - delta, sfreq)
    denom = 0.5 * (cp + cm)
    if denom < 1e-8:
        return float("inf") if c0 > 1e-8 else 0.0
    return float(c0 / denom)


def classify_ssvep(
    eeg_window: np.ndarray,
    freqs: tuple[float, ...],
    sfreq: float,
    delta: float = NCCA_DELTA_HZ,
) -> tuple[float, dict[float, float], dict[float, float]]:
    """Return winner, HUD-safe scores (capped), and raw NCCA (for thresholding)."""
    scores_raw: dict[float, float] = {}
    scores_hud: dict[float, float] = {}
    for f in freqs:
        v = compute_ncca(eeg_window, f, sfreq, delta=delta)
        scores_raw[f] = v
        if np.isinf(v) and v > 0:
            scores_hud[f] = 99.99
        elif not np.isfinite(v):
            scores_hud[f] = 0.0
        else:
            scores_hud[f] = float(min(v, 99.99))
    best = max(freqs, key=lambda fr: (scores_raw[fr] if np.isfinite(scores_raw[fr]) else float("inf")))
    return best, scores_hud, scores_raw


# =============================================================================
# EEG thread + BrainFlow
# =============================================================================


class LiveMetrics:
    """Thread-safe NCCA summary for HUD."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.scores: dict[float, float] = {}
        self.winner: float | None = None
        self.quality_ok: bool = False
        self.note: str = ""

    def update(self, scores: dict[float, float], winner: float, quality_ok: bool, note: str) -> None:
        with self.lock:
            self.scores = dict(scores)
            self.winner = winner
            self.quality_ok = quality_ok
            self.note = note[:200]

    def label(self) -> str:
        with self.lock:
            if not self.scores:
                return self.note or "…"
            parts = [f"NCCA {f:.0f}Hz: {self.scores.get(f, float('nan')):.2f}" for f in sorted(self.scores)]
            line1 = "  |  ".join(parts)
            w = self.winner
            qw = "OK" if self.quality_ok else "LOW"
            if w is not None and np.isfinite(self.scores.get(w, float("nan"))):
                line2 = f"→ DETECTED: {w:.0f} Hz  (quality {qw}, threshold {NCCA_QUALITY_THRESHOLD:.1f})"
            else:
                line2 = f"→ DETECTED: —  (quality {qw})"
            return f"{line1}\n{line2}\n{self.note}"


def _eeg_loop(
    stream: BrainFlowStream,
    freqs: tuple[float, float],
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
        winner, scores_hud, scores_raw = classify_ssvep(eeg_f, freqs, fs, delta=NCCA_DELTA_HZ)

        w_raw = scores_raw.get(winner, 0.0)
        if np.isnan(w_raw):
            quality_ok = False
        elif np.isinf(w_raw) and w_raw > 0:
            quality_ok = True
        else:
            quality_ok = w_raw > NCCA_QUALITY_THRESHOLD

        note = f"causal SOS 0.5–32 Hz + 50 Hz notch  |  win={win_n} @ {fs:.0f} Hz"
        metrics.update(scores_hud, winner, quality_ok, note)


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
        description="PsychoPy dual-target SSVEP + live NCCA (SciPy filters, sklearn CCA).",
    )
    parser.add_argument("--serial-port", default="COM4")
    parser.add_argument(
        "--flicker-hz",
        type=float,
        default=12.0,
        help="Reserved for CLI compatibility; stimuli use fixed 10 Hz and 15 Hz.",
    )
    parser.add_argument(
        "--monitor-hz",
        type=float,
        default=MONITOR_HZ_FALLBACK,
        help="Fallback dt (s) if frame clock yields non-positive delta.",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true", help="Stimulus without BrainFlow.")

    args = parser.parse_args()
    _ = args.flicker_hz  # kept for argparse compatibility (fixed 10/15 Hz stimuli)

    f_lo, f_hi = SSVEP_FREQS

    stop_evt = threading.Event()
    metrics = LiveMetrics()
    stream: BrainFlowStream | None = None
    tid: threading.Thread | None = None

    if not args.no_eeg:
        stream = _prepare_knight(args.serial_port, args.num_channels)
        tid = threading.Thread(
            target=_eeg_loop,
            kwargs={
                "stream": stream,
                "freqs": SSVEP_FREQS,
                "metrics": metrics,
                "stop_evt": stop_evt,
            },
            daemon=True,
        )
        tid.start()

    win = visual.Window(
        fullscr=args.fullscreen,
        color=(-1, -1, -1),
        units="norm",
        waitBlanking=True,
    )

    stim_left = visual.Rect(win, width=0.36, height=0.34, pos=(-0.42, 0.06))
    stim_right = visual.Rect(win, width=0.36, height=0.34, pos=(0.42, 0.06))

    title = visual.TextStim(
        win,
        text="Look at LEFT (10 Hz) or RIGHT (15 Hz) target",
        height=0.055,
        pos=(0.0, 0.82),
        color=(0.85, 0.85, 0.85),
    )
    lab_left = visual.TextStim(
        win, text=f"{f_lo:.0f} Hz", height=0.06, pos=(-0.42, -0.22), color=(0.4, 0.75, 1.0)
    )
    lab_right = visual.TextStim(
        win, text=f"{f_hi:.0f} Hz", height=0.06, pos=(0.42, -0.22), color=(1.0, 0.55, 0.45)
    )
    hint = visual.TextStim(
        win,
        text=f"Nominal display {args.monitor_hz:.0f} Hz  ·  Escape to quit",
        height=0.04,
        pos=(0.0, -0.88),
        color=(0.5, 0.5, 0.5),
    )
    status = visual.TextStim(win, text="", height=0.032, pos=(0.0, -0.52), wrapWidth=1.85)

    if args.no_eeg:
        status.text = "(EEG offline — flicker demo only)"

    clock = core.Clock()
    phase_left = 0.0
    phase_right = 0.0
    clock.reset()

    try:
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                break

            dt = clock.getTime()
            clock.reset()
            dt = dt if dt > 0 else 1.0 / args.monitor_hz
            phase_left = (phase_left + f_lo * dt) % 1.0
            phase_right = (phase_right + f_hi * dt) % 1.0

            stim_left.fillColor = (1, 1, 1) if phase_left < 0.5 else (-1, -1, -1)
            stim_right.fillColor = (1, 1, 1) if phase_right < 0.5 else (-1, -1, -1)

            if not args.no_eeg:
                status.text = metrics.label()

            title.draw()
            lab_left.draw()
            lab_right.draw()
            stim_left.draw()
            stim_right.draw()
            status.draw()
            hint.draw()
            win.flip()
    finally:
        stop_evt.set()
        win.close()
        core.quit()

        if tid is not None:
            tid.join(timeout=WINDOW_SECONDS)

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
