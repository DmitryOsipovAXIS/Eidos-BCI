"""SSVEP collection with a start menu and single/alternating modes."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pygame
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig
from pipeline.realtime_pipeline import RealtimeCCAPipeline
from ws.websockets import start as ws_start, broadcast as ws_broadcast

DEFAULT_MONITOR_HZ = 60.0
DEFAULT_TARGET_HZ = 7.5
DEFAULT_DURATION_S = 120.0
DEFAULT_DISCARD_S = 1.5

DEFAULT_LEFT_HZ = 7.5
DEFAULT_RIGHT_HZ = 12.0
DEFAULT_ALT_TOTAL_S = 10 * 60.0
DEFAULT_ALT_BLOCK_S = 30.0
DEFAULT_CUE_S = 1.0

BG = (0, 0, 0)
WHITE = (255, 255, 255)
DIM = (20, 20, 20)
GREY = (180, 180, 180)
BLUE = (80, 120, 255)
RED = (255, 80, 80)


def _set_mode(size: tuple[int, int], fullscreen: bool) -> pygame.Surface:
    flags = pygame.SCALED
    if fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        return pygame.display.set_mode(size, flags, vsync=0)
    except TypeError:
        return pygame.display.set_mode(size, flags)


def _record_window(
    stream,
    label: int,
    duration_s: float,
    discard_s: float,
    start_event: threading.Event,
    result: dict,
) -> None:
    from acquisition.brainflow_stream import collect_labeled_window
    try:
        start_event.wait(timeout=duration_s + 2)
        eeg, lbl = collect_labeled_window(stream, label, duration_s, discard_s)
        result["eeg"] = eeg
        result["label"] = lbl
    except Exception as e:
        result["error"] = str(e)


def _compute_psd(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(2048, eeg.shape[1])
    psds = []
    freqs = None
    for ch in range(eeg.shape[0]):
        freqs, pxx = welch(eeg[ch], fs=fs, nperseg=nperseg)
        psds.append(pxx)
    mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
    return freqs, mean_psd


def _draw_text_center(screen: pygame.Surface, font: pygame.font.Font, text: str, y: int, color: tuple) -> None:
    surf = font.render(text, True, color)
    screen.blit(surf, surf.get_rect(center=(screen.get_width() // 2, y)))


def _draw_inference_box(
    screen: pygame.Surface,
    fmed: pygame.font.Font,
    fsml: pygame.font.Font,
    label: str,
    scores: list[float],
    peak_hz: float,
) -> None:
    W, H = screen.get_width(), screen.get_height()
    bw, bh = 230, 90
    bx, by = W - bw - 14, H - bh - 14
    pygame.draw.rect(screen, (15, 15, 15), (bx, by, bw, bh), border_radius=8)
    pygame.draw.rect(screen, (90, 90, 90), (bx, by, bw, bh), 2, border_radius=8)

    col = BLUE if label == "LEFT" else (RED if label == "RIGHT" else GREY)
    s = fmed.render(label, True, col)
    screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 24)))

    hz_text = f"{peak_hz:.1f} Hz" if peak_hz > 0 else "---"
    s = fsml.render(hz_text, True, GREY)
    screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 52)))

    if scores:
        parts = [f"L:{scores[0]:.3f}", f"R:{scores[1]:.3f}"] if len(scores) >= 2 else []
        s = fsml.render("  ".join(parts), True, (110, 110, 110))
        screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 74)))


def _draw_single(
    screen: pygame.Surface,
    rect: pygame.Rect,
    on: bool,
    label: int,
    hz: float,
    fbig: pygame.font.Font,
    fmed: pygame.font.Font,
    fps: float,
) -> None:
    col = BLUE if label == 0 else RED
    screen.fill(BG)
    pygame.draw.rect(screen, WHITE if on else DIM, rect)
    pygame.draw.rect(screen, col, rect, 4)
    _draw_text_center(screen, fbig, "FOCUS", rect.top - 40, col)
    _draw_text_center(screen, fmed, f"{hz:0.1f} Hz", rect.bottom + 22, GREY)
    screen.blit(fmed.render(f"FPS {fps:0.1f}", True, GREY), fmed.render(f"FPS {fps:0.1f}", True, GREY).get_rect(center=(screen.get_width() - 90, 26)))


def _draw_dual(
    screen: pygame.Surface,
    lrect: pygame.Rect,
    rrect: pygame.Rect,
    left_on: bool,
    right_on: bool,
    label: int,
    left_hz: float,
    right_hz: float,
    fbig: pygame.font.Font,
    fmed: pygame.font.Font,
    fps: float,
    block_left_s: float,
) -> None:
    screen.fill(BG)
    pygame.draw.rect(screen, WHITE if left_on else DIM, lrect)
    pygame.draw.rect(screen, WHITE if right_on else DIM, rrect)
    pygame.draw.rect(screen, BLUE if label == 0 else (40, 40, 40), lrect, 4)
    pygame.draw.rect(screen, RED if label == 1 else (40, 40, 40), rrect, 4)
    _draw_text_center(screen, fbig, "LOOK LEFT" if label == 0 else "LOOK RIGHT", 60, BLUE if label == 0 else RED)
    _draw_text_center(screen, fmed, f"Left {left_hz:0.1f} Hz", lrect.bottom + 22, BLUE)
    _draw_text_center(screen, fmed, f"Right {right_hz:0.1f} Hz", rrect.bottom + 22, RED)
    _draw_text_center(screen, fmed, f"Block {block_left_s:0.1f}s", 100, GREY)
    screen.blit(fmed.render(f"FPS {fps:0.1f}", True, GREY), fmed.render(f"FPS {fps:0.1f}", True, GREY).get_rect(center=(screen.get_width() - 90, 26)))


def _start_menu(fullscreen: bool) -> dict:
    pygame.init()
    info = pygame.display.Info()
    W, H = (info.current_w, info.current_h) if fullscreen else (int(info.current_w * 0.9), int(info.current_h * 0.9))
    screen = _set_mode((W, H), fullscreen)
    pygame.display.set_caption("SSVEP Start Menu")
    clock = pygame.time.Clock()

    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))
    fsml = pygame.font.SysFont("Arial", max(18, H // 36))

    options = [
        ("1", "Left 7.5 Hz for 2 minutes", {"mode": "single", "side": "left", "hz": 7.5, "duration": 120.0}),
        ("2", "Right 12 Hz for 2 minutes", {"mode": "single", "side": "right", "hz": 12.0, "duration": 120.0}),
        ("3", "Alternate 7.5/12 Hz for 10 minutes", {"mode": "alt", "left_hz": 7.5, "right_hz": 12.0, "alt_total": 600.0, "alt_block": 30.0}),
        ("4", "Alternate BOTH flickers (7.5/12) for 10 minutes", {"mode": "alt", "left_hz": 7.5, "right_hz": 12.0, "alt_total": 600.0, "alt_block": 30.0, "both_flicker": True}),
        ("5", "Use CLI settings (single or alt)", {"mode": "cli"}),
    ]

    selected = None
    option_rects: list[tuple[pygame.Rect, dict]] = []
    while selected is None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit(0)
            if e.type == pygame.KEYDOWN:
                key = pygame.key.name(e.key)
                for k, _, payload in options:
                    if key == k:
                        selected = payload
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                for rect, payload in option_rects:
                    if rect.collidepoint(mx, my):
                        selected = payload

        screen.fill(BG)
        _draw_text_center(screen, fbig, "SSVEP START", H // 6, GREY)
        y = H // 3
        option_rects = []
        for k, text, _ in options:
            lbl = f"{k}. {text}"
            surf = fmed.render(lbl, True, GREY)
            rect = surf.get_rect(center=(W // 2, y))
            screen.blit(surf, rect)
            option_rects.append((rect.inflate(40, 16), _))
            y += 46
        _draw_text_center(screen, fsml, "Press 1-5 or click an option (click to focus)", H - 90, GREY)
        _draw_text_center(screen, fsml, "ESC to quit", H - 60, GREY)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return selected


def _run_single(
    args,
    stream,
    fs: Optional[float],
    side: str,
    hz: float,
    duration_s: float,
    label: int,
    rt_pipeline: Optional[RealtimeCCAPipeline],
) -> Optional[np.ndarray]:
    pygame.init()
    info = pygame.display.Info()
    W, H = (info.current_w, info.current_h) if args.fullscreen else (int(info.current_w * 0.9), int(info.current_h * 0.9))
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP Single Target")
    clock = pygame.time.Clock()

    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))
    fsml = pygame.font.SysFont("Arial", max(18, H // 36))

    bw, bh = int(W * 0.35), int(H * 0.50)
    cy = H // 2
    rect = pygame.Rect(int(W * 0.12) if side == "left" else int(W * 0.53), cy - bh // 2, bw, bh)

    result: dict = {}
    start_event = threading.Event()
    rec_thread = None
    if stream is not None:
        rec_thread = threading.Thread(
            target=_record_window,
            args=(stream, label, duration_s, args.discard, start_event, result),
            daemon=True,
        )
        rec_thread.start()

    phase = 0.0
    step = hz / args.monitor_hz
    end_t = pygame.time.get_ticks() + int(duration_s * 1000)
    started = False
    running = True
    while pygame.time.get_ticks() < end_t and running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        phase = (phase + step) % 1.0
        _draw_single(screen, rect, phase < 0.5, label, hz, fbig, fmed, clock.get_fps())

        if rt_pipeline is not None:
            rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
            _draw_inference_box(screen, fmed, fsml, rt_label, rt_scores, rt_peak_hz)

        pygame.display.flip()
        if stream is not None and not started:
            start_event.set()
            started = True
        clock.tick(args.monitor_hz)

    pygame.quit()
    if stream is not None and rec_thread is not None:
        rec_thread.join(timeout=duration_s + 5)
    if "error" in result:
        print(f"Recording error: {result['error']}", flush=True)
        return None
    return result.get("eeg")


def _run_alternating(
    args,
    stream,
    fs: Optional[float],
    left_hz: float,
    right_hz: float,
    alt_total_s: float,
    alt_block_s: float,
    label_start: int,
    rt_pipeline: Optional[RealtimeCCAPipeline],
    both_flicker: bool,
) -> list[np.ndarray]:
    pygame.init()
    info = pygame.display.Info()
    W, H = (info.current_w, info.current_h) if args.fullscreen else (int(info.current_w * 0.9), int(info.current_h * 0.9))
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP Alternating")
    clock = pygame.time.Clock()

    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))
    fsml = pygame.font.SysFont("Arial", max(18, H // 36))

    bw, bh = int(W * 0.32), int(H * 0.46)
    cy = H // 2
    lrect = pygame.Rect(int(W * 0.08), cy - bh // 2, bw, bh)
    rrect = pygame.Rect(int(W * 0.60), cy - bh // 2, bw, bh)

    left_step = left_hz / args.monitor_hz
    right_step = right_hz / args.monitor_hz
    blocks = max(1, int(alt_total_s / alt_block_s))
    labels = [(label_start + i) % 2 for i in range(blocks)]
    results: list[np.ndarray] = []
    left_phase = right_phase = 0.0

    for idx, label in enumerate(labels):
        result: dict = {}
        start_event = threading.Event()
        rec_thread = None
        if stream is not None:
            rec_thread = threading.Thread(
                target=_record_window,
                args=(stream, label, alt_block_s, args.discard, start_event, result),
                daemon=True,
            )
            rec_thread.start()

        end_t = pygame.time.get_ticks() + int(alt_block_s * 1000)
        started = False
        running = True
        while pygame.time.get_ticks() < end_t and running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False

            left_phase = (left_phase + left_step) % 1.0
            right_phase = (right_phase + right_step) % 1.0
            block_left = max(0.0, (end_t - pygame.time.get_ticks()) / 1000.0)
            if both_flicker:
                left_on = left_phase < 0.5
                right_on = right_phase < 0.5
            else:
                left_on = left_phase < 0.5 if label == 0 else False
                right_on = right_phase < 0.5 if label == 1 else False
            _draw_dual(
                screen, lrect, rrect,
                left_on,
                right_on,
                label, left_hz, right_hz,
                fbig, fmed, clock.get_fps(), block_left,
            )

            if rt_pipeline is not None:
                rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
                _draw_inference_box(screen, fmed, fsml, rt_label, rt_scores, rt_peak_hz)

            pygame.display.flip()
            if stream is not None and not started:
                start_event.set()
                started = True
            clock.tick(args.monitor_hz)

        if rec_thread is not None:
            rec_thread.join(timeout=alt_block_s + 5)
        if "error" in result:
            print(f"Block {idx + 1} error: {result['error']}", flush=True)
        elif "eeg" in result:
            results.append(result["eeg"])

        if not running:
            break

    pygame.quit()
    return results


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

    fig, axes = plt.subplots(n_ch, 2, figsize=(14, max(6, n_ch * 1.8)), sharex="col")
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


def _print_psd_summary(eeg: np.ndarray, fs: float, target_hz: float) -> None:
    freqs, mean_psd = _compute_psd(eeg, fs)
    mask = (freqs >= target_hz - 2.0) & (freqs <= target_hz + 2.0)
    peak_freq = float(freqs[mask][np.argmax(mean_psd[mask])]) if np.any(mask) else float(freqs[np.argmax(mean_psd)])
    peak_power = float(np.max(mean_psd[mask])) if np.any(mask) else float(np.max(mean_psd))
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    print(f"Recorded PSD shape={mean_psd.shape} (mean over channels)", flush=True)
    print(f"Peak freq in target band={peak_freq:0.3f} Hz (power={peak_power:0.3g})", flush=True)
    print(f"Power at {target_hz:0.3f} Hz -> bin {freqs[idx]:0.3f} Hz, power={mean_psd[idx]:0.3g}", flush=True)


def _start_ws_server() -> asyncio.AbstractEventLoop:
    ws_loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_until_complete(ws_start())

    threading.Thread(target=_run, daemon=True).start()
    print("WebSocket server started on ws://localhost:8765", flush=True)
    return ws_loop


def _init_board(args) -> tuple[Optional[object], Optional[float]]:
    from acquisition.brainflow_stream import BrainFlowStream
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

    BoardShim.disable_board_logger()
    p = BrainFlowInputParams()
    if args.serial_port:
        p.serial_port = args.serial_port
    stream = BrainFlowStream(
        board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
        params=p,
        num_channels=args.num_channels,
    )
    stream.prepare_session()
    stream.start_stream()
    time.sleep(2)
    for ch in range(1, args.num_channels + 1):
        time.sleep(0.25)
        try:
            stream.board.config_board(f"chon_{ch}_12")
        except Exception:
            pass
        time.sleep(0.25)
        try:
            stream.board.config_board(f"rldadd_{ch}")
        except Exception:
            pass
        time.sleep(0.25)
    fs = stream.sampling_rate()
    if abs(fs - 125.0) > 1e-6:
        stream.stop_stream()
        stream.release_session()
        raise RuntimeError(f"Board sampling rate is {fs:0.3f} Hz, expected 125.000 Hz")
    print(f"Board ready at {fs:0.3f} Hz", flush=True)
    return stream, fs


def main() -> None:
    parser = argparse.ArgumentParser(description="SSVEP collection with start menu")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--monitor-hz", type=float, default=DEFAULT_MONITOR_HZ)
    parser.add_argument("--discard", type=float, default=DEFAULT_DISCARD_S)
    parser.add_argument("--mode", choices=["menu", "single", "alt"], default="menu")
    parser.add_argument("--single-hz", type=float, default=DEFAULT_TARGET_HZ)
    parser.add_argument("--single-side", choices=["left", "right"], default="left")
    parser.add_argument("--single-duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--left-hz", type=float, default=DEFAULT_LEFT_HZ)
    parser.add_argument("--right-hz", type=float, default=DEFAULT_RIGHT_HZ)
    parser.add_argument("--alt-total", type=float, default=DEFAULT_ALT_TOTAL_S)
    parser.add_argument("--alt-block", type=float, default=DEFAULT_ALT_BLOCK_S)
    parser.add_argument("--alt-start", type=int, choices=[0, 1], default=0)
    parser.add_argument("--both-flicker", action="store_true",
        help="During alternating blocks, flicker both targets while cueing focus")
    parser.add_argument("--save", action="store_true", help="Save raw X/y to data/raw")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--confidence-ratio", type=float, default=1.3)
    parser.add_argument("--min-score", type=float, default=0.02)
    args = parser.parse_args()

    stream = None
    fs: Optional[float] = None
    if args.mode == "menu" and not args.no_eeg:
        print("Initializing board...", flush=True)
        stream, fs = _init_board(args)

    selection = None
    if args.mode == "menu":
        selection = _start_menu(args.fullscreen)
        if selection.get("mode") != "cli":
            args.mode = selection["mode"]

    if args.mode == "single":
        side = selection.get("side", args.single_side) if selection else args.single_side
        hz = selection.get("hz", args.single_hz) if selection else args.single_hz
        duration_s = selection.get("duration", args.single_duration) if selection else args.single_duration
        label = 0 if side == "left" else 1
        frequencies_hz = [args.left_hz, args.right_hz]
        print("collecten: single-target test", flush=True)
        print(f"duration={duration_s:0.1f}s, target={hz:0.2f} Hz, label={label}", flush=True)
    elif args.mode == "alt":
        left_hz = selection.get("left_hz", args.left_hz) if selection else args.left_hz
        right_hz = selection.get("right_hz", args.right_hz) if selection else args.right_hz
        alt_total_s = selection.get("alt_total", args.alt_total) if selection else args.alt_total
        alt_block_s = selection.get("alt_block", args.alt_block) if selection else args.alt_block
        both_flicker = selection.get("both_flicker", args.both_flicker) if selection else args.both_flicker
        frequencies_hz = [left_hz, right_hz]
        print("collecten: alternating test", flush=True)
        print(f"total={alt_total_s:0.1f}s, block={alt_block_s:0.1f}s, left={left_hz:0.2f} Hz, right={right_hz:0.2f} Hz", flush=True)
    else:
        print("collecten: using CLI settings", flush=True)
        side = args.single_side
        hz = args.single_hz
        duration_s = args.single_duration
        label = 0 if side == "left" else 1
        frequencies_hz = [args.left_hz, args.right_hz]
        both_flicker = args.both_flicker

    if stream is None and not args.no_eeg:
        stream, fs = _init_board(args)

    ws_loop = _start_ws_server()

    rt_pipeline: Optional[RealtimeCCAPipeline] = None
    if stream is not None:
        rt_pipeline = RealtimeCCAPipeline(
            stream=stream,
            frequencies_hz=frequencies_hz,
            sample_rate_hz=fs or 125.0,
            window_s=args.window_s,
            step_s=args.step_s,
            confidence_ratio=args.confidence_ratio,
            min_absolute=args.min_score,
            ws_loop=ws_loop,
            ws_broadcast=ws_broadcast,
        )
        rt_pipeline.start()
        print(f"Real-time CCA pipeline started (window={args.window_s}s, step={args.step_s}s)", flush=True)

    if args.mode == "alt":
        eeg_list = _run_alternating(
            args, stream, fs,
            left_hz, right_hz, alt_total_s, alt_block_s, args.alt_start,
            rt_pipeline,
            both_flicker,
        )
        if rt_pipeline is not None:
            rt_pipeline.stop()
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            stream.release_session()

        all_guesses = rt_pipeline.get_full_history() if rt_pipeline is not None else []
        if not eeg_list:
            print("No EEG recorded.", flush=True)
            _show_report(None, fs or 125.0, frequencies_hz, all_guesses)
            return
        config = SSVEPConfig(left_hz=left_hz, right_hz=right_hz)
        config.data_raw_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        min_len = min(x.shape[1] for x in eeg_list)
        X = np.stack([x[:, :min_len] for x in eeg_list])
        y = np.array([(args.alt_start + i) % 2 for i in range(len(eeg_list))], dtype=np.int64)
        x_path = config.data_raw_dir / f"X_{ts}.npy"
        y_path = config.data_raw_dir / f"y_{ts}.npy"
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"Saved raw data -> {x_path}  {y_path}", flush=True)
        _show_report(eeg_list[0], fs or 125.0, frequencies_hz, all_guesses)
        return

    eeg = _run_single(
        args, stream, fs, side, hz, duration_s, label,
        rt_pipeline,
    )
    if rt_pipeline is not None:
        rt_pipeline.stop()
    if stream is not None:
        try:
            stream.stop_stream()
        except Exception:
            pass
        stream.release_session()

    if eeg is None:
        print("No EEG recorded.", flush=True)
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

    _show_report(eeg, fs or 125.0, frequencies_hz, all_guesses)


if __name__ == "__main__":
    main()
