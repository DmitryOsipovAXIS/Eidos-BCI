"""Single-target SSVEP: PsychoPy flicker + block evaluation (3 × 10 s windows).

One flickering stimulus; EEG is collected in three non-overlapping 10-second windows.
Each window is filtered (occipital channels) and classified with harmonic-weighted CCA
across candidate frequencies. After the third window, results are aggregated by
majority vote (confidence tie-break) and printed.

Usage (from ``bci-ssvep``):

  python scripts/ssvep_psychopy_live.py --serial-port COM4
  python scripts/ssvep_psychopy_live.py --fullscreen --flicker-hz 12
  python scripts/ssvep_psychopy_live.py --candidate-freqs 7.5,10,12,15
  python scripts/ssvep_psychopy_live.py --no-eeg

Runs exactly one block (3 windows), then freezes on the final result until Escape.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

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

# -----------------------------------------------------------------------------
# Block evaluation parameters
# -----------------------------------------------------------------------------

DEFAULT_FLICKER_HZ = 7.5
N_WINDOWS = 3
WINDOW_DURATION_S = 10.0
MONITOR_HZ_FALLBACK = 60.0
N_HARMONICS = 3
PEAK_SEARCH_HZ = (5.0, 29.0)
FBCCA_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (6.0, 90.0),
    (14.0, 90.0),
    (22.0, 90.0),
    (30.0, 90.0),
    (38.0, 90.0),
)

DEFAULT_CANDIDATE_FREQS: tuple[float, ...] = (7.5, 10.0, 12.0, 15.0)

_KNIGHT_CH_NAMES: tuple[str, ...] = (
    "PO3",
    "POz",
    "PO4",
    "PO7",
    "O7",
    "Oz",
    "O2",
    "PO8",
)
_OCCIPITAL_PREFERRED: tuple[str, ...] = ("O1", "Oz", "O2")

BG_GRAY = (-0.6, -0.6, -0.6)
STIM_SIZE = 0.72


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class WindowResult:
    index: int
    winner_hz: float
    score: float
    scores: dict[float, float]
    peak_hz: float | None


@dataclass
class BlockResult:
    windows: list[WindowResult] = field(default_factory=list)
    dominant_hz: float | None = None
    consistency: str = "LOW"


# =============================================================================
# Occipital channels + filtering
# =============================================================================


def occipital_row_indices(num_channels: int) -> list[int]:
    names = list(_KNIGHT_CH_NAMES[: max(1, min(num_channels, len(_KNIGHT_CH_NAMES)))])
    rows: list[int] = []
    for label in _OCCIPITAL_PREFERRED:
        if label in names:
            rows.append(names.index(label))
    if "O1" not in names and "O7" in names and names.index("O7") not in rows:
        rows.insert(0, names.index("O7"))
    if not rows:
        rows = list(range(min(num_channels, 3)))
    return rows


def design_filter_sos(fs: float) -> tuple[np.ndarray, np.ndarray]:
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


def filter_eeg_window(eeg: np.ndarray, sos_notch: np.ndarray, sos_band: np.ndarray) -> np.ndarray:
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2-D (channels x samples).")
    x = sosfilt(sos_notch, eeg, axis=-1)
    return sosfilt(sos_band, x, axis=-1)


def welch_peak_hz(eeg_f: np.ndarray, fs: float) -> float | None:
    sig = np.mean(eeg_f, axis=0)
    n_seg = int(min(fs * 2, sig.shape[-1]))
    n_seg = max(128, n_seg - (n_seg % 2))
    fq, pw = scipy_welch(sig, fs=fs, nperseg=n_seg, noverlap=n_seg // 2)
    lo, hi = PEAK_SEARCH_HZ
    m = (fq >= lo) & (fq <= hi)
    if not np.any(m):
        return None
    jj = int(np.argmax(pw[m]))
    return float(fq[m][jj])


# =============================================================================
# Harmonic-weighted CCA (multi-frequency)
# =============================================================================


def _reference_one_harmonic(
    freq: float, sfreq: float, n_samples: int, harmonic: int
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    ph = 2.0 * np.pi * freq * harmonic * t
    return np.stack([np.sin(ph), np.cos(ph)], axis=1)


def _cca_correlation(eeg: np.ndarray, y_ref: np.ndarray) -> float:
    _, n_s = eeg.shape
    if n_s < 8 or y_ref.shape[0] != n_s:
        return 0.0
    x_eeg = eeg.T.copy()
    x_eeg -= x_eeg.mean(axis=0, keepdims=True)
    y = y_ref - y_ref.mean(axis=0, keepdims=True)
    if x_eeg.shape[1] < 1 or y.shape[1] < 1:
        return 0.0
    cca = CCA(n_components=1, max_iter=800)
    try:
        cca.fit(x_eeg, y)
        x_c, y_c = cca.transform(x_eeg, y)
        r = np.corrcoef(x_c.ravel(), y_c.ravel())[0, 1]
        if not np.isfinite(r):
            return 0.0
        return float(np.clip(abs(r), 0.0, 1.0))
    except Exception:
        return 0.0


def harmonic_weighted_cca(
    eeg: np.ndarray,
    freq: float,
    sfreq: float,
    n_harmonics: int = N_HARMONICS,
) -> float:
    weights = np.array([1.0 / h for h in range(1, n_harmonics + 1)], dtype=np.float64)
    scores = np.zeros(n_harmonics, dtype=np.float64)
    _, n_s = eeg.shape
    for i, h in enumerate(range(1, n_harmonics + 1)):
        y_ref = _reference_one_harmonic(freq, sfreq, n_s, h)
        scores[i] = _cca_correlation(eeg, y_ref)
    return float(np.dot(weights, scores) / weights.sum())


def _design_fbcca_band_sos(fs: float, lo_hz: float, hi_hz: float) -> np.ndarray | None:
    """Design one FBCCA bandpass filter, clipped to valid Nyquist range."""
    nyq = fs / 2.0
    lo = max(lo_hz / nyq, 1e-5)
    hi = min(hi_hz / nyq, 1.0 - 1e-5)
    if hi <= lo:
        return None
    return butter(N=4, Wn=[lo, hi], btype="band", output="sos")


def fbcca_classify_window(
    eeg_window: np.ndarray,
    candidates: tuple[float, ...],
    sfreq: float,
) -> tuple[float, float, dict[float, float]]:
    """
    Perform FBCCA classification over multiple filter banks.

    Returns:
        winner_frequency, best_score, fused_scores_dict
    """
    n_bands = len(FBCCA_BANDS_HZ)
    weights = np.array([1.0 / (i + 1) for i in range(n_bands)], dtype=np.float64)
    weights /= weights.sum()
    fused_scores = {f: 0.0 for f in candidates}
    used_weight = 0.0

    for i, (lo_hz, hi_hz) in enumerate(FBCCA_BANDS_HZ):
        sos_fb = _design_fbcca_band_sos(sfreq, lo_hz, hi_hz)
        if sos_fb is None:
            continue

        eeg_fb = sosfilt(sos_fb, eeg_window, axis=-1)
        band_scores = {f: harmonic_weighted_cca(eeg_fb, f, sfreq) for f in candidates}

        # Per-band z-score keeps noisiest bands from dominating the weighted sum.
        vals = np.array(list(band_scores.values()), dtype=np.float64)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma > 1e-8:
            band_scores = {f: float((v - mu) / sigma) for f, v in band_scores.items()}

        wi = float(weights[i])
        used_weight += wi
        for f in candidates:
            fused_scores[f] += wi * band_scores[f]

    if used_weight <= 0.0:
        # Fallback for invalid filter banks at very low sample rates.
        return classify_window(eeg_window, candidates, sfreq, use_fbcca=False)

    if used_weight < 1.0:
        scale = 1.0 / used_weight
        for f in candidates:
            fused_scores[f] *= scale

    winner = max(candidates, key=lambda f: fused_scores[f])
    return winner, fused_scores[winner], fused_scores


def classify_window(
    eeg_window: np.ndarray,
    candidates: tuple[float, ...],
    sfreq: float,
    *,
    use_fbcca: bool = False,
) -> tuple[float, float, dict[float, float]]:
    if use_fbcca:
        return fbcca_classify_window(eeg_window, candidates, sfreq)
    scores = {f: harmonic_weighted_cca(eeg_window, f, sfreq) for f in candidates}
    winner = max(candidates, key=lambda f: scores[f])
    return winner, scores[winner], scores


def aggregate_block(windows: list[WindowResult]) -> BlockResult:
    if not windows:
        return BlockResult()

    votes = Counter(w.winner_hz for w in windows)
    top_count = votes.most_common(1)[0][1]
    leaders = [f for f, c in votes.items() if c == top_count]

    if len(leaders) == 1:
        dominant = leaders[0]
    else:
        mean_scores: dict[float, float] = {}
        for f in leaders:
            vals = [w.scores[f] for w in windows]
            mean_scores[f] = float(np.mean(vals))
        dominant = max(leaders, key=lambda f: mean_scores[f])

    agree = sum(1 for w in windows if w.winner_hz == dominant)
    if agree == N_WINDOWS:
        consistency = "HIGH"
    elif agree >= 2:
        consistency = "MEDIUM"
    else:
        consistency = "LOW"

    return BlockResult(windows=windows, dominant_hz=dominant, consistency=consistency)


def print_block_result(block: BlockResult) -> None:
    lines = ["", "===== SSVEP RESULT ====="]
    for w in block.windows:
        peak = "—" if w.peak_hz is None else f"{w.peak_hz:.1f}"
        lines.append(f"Window {w.index}: f = {w.winner_hz:g} Hz (score {w.score:.3f}, PSD peak {peak} Hz)")
    dom = block.dominant_hz
    dom_s = f"{dom:g} Hz" if dom is not None else "—"
    lines.append("")
    lines.append(f"FINAL DOMINANT FREQUENCY: {dom_s}")
    lines.append(f"Confidence: {block.consistency}")
    lines.append("=" * 24)
    lines.append("")
    print("\n".join(lines), flush=True)


def parse_candidate_freqs(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --candidate-freqs")
    return tuple(sorted({float(p) for p in parts}))


def resolve_candidates(flicker_hz: float, candidate_arg: str | None) -> tuple[float, ...]:
    if candidate_arg:
        base = set(parse_candidate_freqs(candidate_arg))
    else:
        base = set(DEFAULT_CANDIDATE_FREQS)
    base.add(flicker_hz)
    return tuple(sorted(base))


# =============================================================================
# Shared UI state (minimal, updated only between windows)
# =============================================================================


class BlockMetrics:
    def __init__(self, n_windows: int = N_WINDOWS) -> None:
        self.lock = threading.Lock()
        self.experiment_finished = threading.Event()
        self.n_windows = n_windows
        self.phase: str = "waiting"
        self.window_index: int = 0
        self.seconds_left: float = WINDOW_DURATION_S
        self.last_window: WindowResult | None = None
        self.current_block: BlockResult | None = None

    def is_finished(self) -> bool:
        return self.experiment_finished.is_set()

    def set_acquiring(self, window_index: int, seconds_left: float) -> None:
        with self.lock:
            if self.experiment_finished.is_set():
                return
            self.phase = "acquiring"
            self.window_index = window_index
            self.seconds_left = max(0.0, seconds_left)

    def set_window_done(self, result: WindowResult) -> None:
        with self.lock:
            if self.experiment_finished.is_set():
                return
            self.phase = "window_done"
            self.last_window = result

    def finish_experiment(self, block: BlockResult) -> None:
        with self.lock:
            self.phase = "finished"
            self.current_block = block
        self.experiment_finished.set()

    def hud_text(self) -> str:
        with self.lock:
            if self.phase == "finished":
                return self._format_final_locked()
            if self.phase == "waiting":
                return "Preparing EEG…"
            if self.phase == "acquiring":
                return (
                    f"Window {self.window_index}/{self.n_windows} — acquiring "
                    f"({self.seconds_left:.0f}s left)"
                )
            if self.phase == "window_done" and self.last_window is not None:
                w = self.last_window
                return f"Window {w.index} done: {w.winner_hz:g} Hz (score {w.score:.2f})"
            return ""

    def _format_final_locked(self) -> str:
        b = self.current_block
        if b is None:
            return "Experiment complete"
        lines = ["===== SSVEP RESULT ====="]
        for w in b.windows:
            lines.append(f"Window {w.index}: {w.winner_hz:g} Hz (score {w.score:.2f})")
        dom = b.dominant_hz
        dom_s = f"{dom:g} Hz" if dom is not None else "—"
        lines.append("")
        lines.append(f"FINAL DOMINANT FREQUENCY: {dom_s}")
        lines.append(f"Confidence: {b.consistency}")
        lines.append("")
        lines.append("Press Escape to quit")
        return "\n".join(lines)


# =============================================================================
# EEG block worker (3 × 10 s, non-overlapping)
# =============================================================================


def _acquire_window_segment(
    stream: BrainFlowStream,
    win_n: int,
    window_index: int,
    metrics: BlockMetrics,
    stop_evt: threading.Event,
) -> np.ndarray | None:
    """Flush at window start, then block until exactly ``win_n`` new samples arrive."""
    stream.get_board_data()
    t0 = time.perf_counter()
    while stream.get_board_data_count() < win_n:
        if stop_evt.is_set() or metrics.is_finished():
            return None
        remaining = WINDOW_DURATION_S - (time.perf_counter() - t0)
        metrics.set_acquiring(window_index, remaining)
        time.sleep(0.2)

    raw = stream.get_board_data()
    eeg_ix = stream.eeg_channel_indices()
    eeg = raw[eeg_ix, :].astype(np.float64)
    if eeg.shape[1] < win_n:
        return None
    return eeg[:, -win_n:]


def _eeg_block_loop(
    stream: BrainFlowStream,
    candidates: tuple[float, ...],
    metrics: BlockMetrics,
    stop_evt: threading.Event,
    *,
    num_channels: int,
    use_fbcca: bool,
) -> None:
    fs = stream.sampling_rate()
    win_n = max(int(WINDOW_DURATION_S * fs), 256)
    occ_rows = occipital_row_indices(num_channels)
    sos_notch, sos_band = design_filter_sos(fs)
    time.sleep(1.0)

    if metrics.is_finished() or stop_evt.is_set():
        return

    window_results: list[WindowResult] = []

    for w_idx in range(1, N_WINDOWS + 1):
        if stop_evt.is_set() or metrics.is_finished():
            return

        segment = _acquire_window_segment(stream, win_n, w_idx, metrics, stop_evt)
        if segment is None:
            return

        eeg_occ = segment[occ_rows, :]
        eeg_f = filter_eeg_window(eeg_occ, sos_notch, sos_band)
        winner, score, scores = classify_window(eeg_f, candidates, fs, use_fbcca=use_fbcca)
        peak = welch_peak_hz(eeg_f, fs)

        result = WindowResult(
            index=w_idx,
            winner_hz=winner,
            score=score,
            scores=scores,
            peak_hz=peak,
        )
        window_results.append(result)
        metrics.set_window_done(result)
        print(f"  Window {w_idx}/{N_WINDOWS}: {winner:g} Hz (score {score:.3f})", flush=True)

    block = aggregate_block(window_results)
    print_block_result(block)
    metrics.finish_experiment(block)


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
    occ = occipital_row_indices(num_channels)
    names = [_KNIGHT_CH_NAMES[i] for i in occ if i < len(_KNIGHT_CH_NAMES)]
    print(f"Board ready. Occipital: {names}", flush=True)
    print(f"Block design: {N_WINDOWS} × {WINDOW_DURATION_S:.0f} s non-overlapping windows", flush=True)
    return stream


# =============================================================================
# PsychoPy stimulus
# =============================================================================


def _flicker_on(frame_index: int, monitor_hz: float, flicker_hz: float) -> bool:
    frames_per_cycle = max(1, round(monitor_hz / flicker_hz))
    half = max(1, frames_per_cycle // 2)
    return (frame_index % frames_per_cycle) < half


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSVEP flicker + 3×10 s block evaluation (harmonic CCA).",
    )
    parser.add_argument("--serial-port", default="COM4")
    parser.add_argument("--flicker-hz", type=float, default=DEFAULT_FLICKER_HZ)
    parser.add_argument(
        "--candidate-freqs",
        default=None,
        help="Comma-separated candidate frequencies for CCA (default includes 7.5,10,12,15 + flicker).",
    )
    parser.add_argument("--monitor-hz", type=float, default=MONITOR_HZ_FALLBACK)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true")
    parser.add_argument(
        "--use-fbcca",
        action="store_true",
        help="Use Filter Bank CCA instead of single-band harmonic CCA.",
    )
    args = parser.parse_args()

    flicker_hz = float(args.flicker_hz)
    candidates = resolve_candidates(flicker_hz, args.candidate_freqs)
    stop_evt = threading.Event()
    metrics = BlockMetrics()
    stream: BrainFlowStream | None = None
    tid_eeg: threading.Thread | None = None

    if not args.no_eeg:
        stream = _prepare_knight(args.serial_port, args.num_channels)
        print(f"Candidates: {', '.join(f'{f:g}' for f in candidates)} Hz", flush=True)
        print(f"Classifier: {'FBCCA' if args.use_fbcca else 'harmonic CCA'}", flush=True)
        tid_eeg = threading.Thread(
            target=_eeg_block_loop,
            kwargs={
                "stream": stream,
                "candidates": candidates,
                "metrics": metrics,
                "stop_evt": stop_evt,
                "num_channels": args.num_channels,
                "use_fbcca": args.use_fbcca,
            },
            daemon=True,
        )
        tid_eeg.start()

    win = visual.Window(
        fullscr=args.fullscreen,
        color=BG_GRAY,
        colorSpace="rgb",
        units="norm",
        waitBlanking=True,
    )

    monitor_hz = float(args.monitor_hz)
    try:
        win.flip()
        core.wait(0.05)
        measured = win.getActualFrameRate(nIdentical=30, nMaxFrames=120, nWarmUpFrames=30)
        if measured and measured > 10:
            monitor_hz = float(measured)
    except Exception:
        pass

    stim = visual.Rect(
        win,
        width=STIM_SIZE,
        height=STIM_SIZE,
        pos=(0.0, 0.0),
        fillColor=(1.0, 1.0, 1.0),
        lineColor=None,
    )
    fix_h = visual.Line(
        win, start=(-0.02, 0.0), end=(0.02, 0.0), lineColor=(0.85, 0.85, 0.85), lineWidth=2
    )
    fix_v = visual.Line(
        win, start=(0.0, -0.02), end=(0.0, 0.02), lineColor=(0.85, 0.85, 0.85), lineWidth=2
    )
    status = visual.TextStim(
        win,
        text="",
        height=0.045,
        pos=(0.0, -0.82),
        color=(0.9, 0.9, 0.9),
        wrapWidth=1.6,
        alignText="center",
    )
    hint = visual.TextStim(
        win,
        text=(
            f"{flicker_hz:g} Hz  ·  {N_WINDOWS}×{WINDOW_DURATION_S:.0f}s evaluation  ·  "
            f"Escape to quit"
        ),
        height=0.035,
        pos=(0.0, 0.88),
        color=(0.55, 0.55, 0.55),
    )

    if args.no_eeg:
        status.text = "EEG offline — flicker only"

    frame_index = 0
    experiment_done = False

    try:
        while True:
            if "escape" in event.getKeys():
                break

            if not args.no_eeg and metrics.is_finished():
                experiment_done = True

            if experiment_done:
                stim.fillColor = BG_GRAY
                status.text = metrics.hud_text()
                hint.text = "Experiment complete — press Escape to quit"
            else:
                stim.fillColor = (
                    (1.0, 1.0, 1.0) if _flicker_on(frame_index, monitor_hz, flicker_hz) else BG_GRAY
                )
                frame_index += 1
                if not args.no_eeg:
                    status.text = metrics.hud_text()

            stim.draw()
            fix_h.draw()
            fix_v.draw()
            status.draw()
            hint.draw()
            win.flip()
    finally:
        stop_evt.set()
        win.close()
        core.quit()

        if tid_eeg is not None:
            tid_eeg.join(timeout=N_WINDOWS * WINDOW_DURATION_S + 5.0)

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
