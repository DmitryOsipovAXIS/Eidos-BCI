"""Route realtime SSVEP guesses to mouse clicks on left/right screen targets.

Usage example:
  python scripts/live_click_router.py --serial-port COM3 --num-channels 8
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path
from typing import Optional

import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acquisition.brainflow_stream import BrainFlowStream
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pipeline.realtime_pipeline import RealtimeCCAPipeline


BG = (15, 15, 20)
PANEL = (28, 28, 36)
TEXT = (220, 220, 230)
BLUE = (70, 120, 255)
RED = (255, 90, 90)
DIM = (90, 90, 100)
GREEN = (90, 220, 140)


class _MouseClicker:
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32

    def screen_size(self) -> tuple[int, int]:
        return int(self.user32.GetSystemMetrics(0)), int(self.user32.GetSystemMetrics(1))

    def click(self, x: int, y: int) -> None:
        self.user32.SetCursorPos(int(x), int(y))
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _init_board(serial_port: Optional[str], num_channels: int) -> tuple[BrainFlowStream, float]:
    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    if serial_port:
        params.serial_port = serial_port

    stream = BrainFlowStream(
        board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
        params=params,
        num_channels=num_channels,
    )
    stream.prepare_session()
    stream.start_stream()
    time.sleep(2.0)

    # Configure channels (same pattern as your existing working scripts)
    for ch in range(1, num_channels + 1):
        time.sleep(0.15)
        try:
            stream.board.config_board(f"chon_{ch}_12")
        except Exception:
            pass
        time.sleep(0.15)
        try:
            stream.board.config_board(f"rldadd_{ch}")
        except Exception:
            pass
        time.sleep(0.15)

    fs = stream.sampling_rate()
    if abs(fs - 125.0) > 1e-6:
        stream.stop_stream()
        stream.release_session()
        raise RuntimeError(f"Board sampling rate is {fs:.3f} Hz, expected 125.000 Hz")
    return stream, fs


def _resolve_positions(args) -> tuple[tuple[int, int], tuple[int, int]]:
    clicker = _MouseClicker()
    sw, sh = clicker.screen_size()

    lx = args.left_x if args.left_x is not None else int(sw * 0.25)
    ly = args.left_y if args.left_y is not None else int(sh * 0.5)
    rx = args.right_x if args.right_x is not None else int(sw * 0.75)
    ry = args.right_y if args.right_y is not None else int(sh * 0.5)
    return (lx, ly), (rx, ry)


def _draw_ui(
    screen: pygame.Surface,
    f_big: pygame.font.Font,
    f_mid: pygame.font.Font,
    f_small: pygame.font.Font,
    left_rect: pygame.Rect,
    right_rect: pygame.Rect,
    left_count: int,
    right_count: int,
    left_hz: float,
    right_hz: float,
    latest_label: str,
    latest_peak_hz: float,
    left_flash_until: float,
    right_flash_until: float,
) -> None:
    now = time.time()
    screen.fill(BG)

    # Counters
    left_count_s = f_big.render(str(left_count), True, BLUE)
    right_count_s = f_big.render(str(right_count), True, RED)
    screen.blit(left_count_s, left_count_s.get_rect(center=(left_rect.centerx, left_rect.top - 46)))
    screen.blit(right_count_s, right_count_s.get_rect(center=(right_rect.centerx, right_rect.top - 46)))

    # Buttons
    l_col = BLUE if now < left_flash_until else DIM
    r_col = RED if now < right_flash_until else DIM
    pygame.draw.rect(screen, l_col, left_rect, border_radius=12)
    pygame.draw.rect(screen, r_col, right_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL, left_rect, 2, border_radius=12)
    pygame.draw.rect(screen, PANEL, right_rect, 2, border_radius=12)

    left_label = f_mid.render(f"LEFT {left_hz:.1f} Hz", True, TEXT)
    right_label = f_mid.render(f"RIGHT {right_hz:.1f} Hz", True, TEXT)
    screen.blit(left_label, left_label.get_rect(center=left_rect.center))
    screen.blit(right_label, right_label.get_rect(center=right_rect.center))

    # Status line
    guess_txt = f"Latest guess: {latest_label}"
    if latest_peak_hz > 0:
        guess_txt += f"   peak: {latest_peak_hz:.2f} Hz"
    st = f_small.render(guess_txt, True, GREEN if latest_label in ("LEFT", "RIGHT") else TEXT)
    screen.blit(st, st.get_rect(center=(screen.get_width() // 2, 36)))

    hint = f_small.render("ESC to stop", True, TEXT)
    screen.blit(hint, hint.get_rect(center=(screen.get_width() // 2, screen.get_height() - 22)))

    pygame.display.flip()


def main() -> None:
    p = argparse.ArgumentParser(description="Realtime SSVEP -> left/right mouse click router")
    p.add_argument("--serial-port", default=None)
    p.add_argument("--num-channels", type=int, default=8)
    p.add_argument("--left-hz", type=float, default=7.5)
    p.add_argument("--right-hz", type=float, default=12.0)

    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--step-s", type=float, default=0.5)
    p.add_argument("--confidence-ratio", type=float, default=1.3)
    p.add_argument("--min-score", type=float, default=0.02)

    p.add_argument("--left-x", type=int, default=None)
    p.add_argument("--left-y", type=int, default=None)
    p.add_argument("--right-x", type=int, default=None)
    p.add_argument("--right-y", type=int, default=None)

    p.add_argument("--cooldown-s", type=float, default=3.0, help="minimum seconds between clicks")
    p.add_argument("--dry-run", action="store_true", help="print clicks but do not click")
    p.add_argument("--ui-width", type=int, default=1100)
    p.add_argument("--ui-height", type=int, default=620)
    p.add_argument("--no-os-click", action="store_true", help="only click in this script UI, do not move/click system mouse")
    p.add_argument("--startup-pause-s", type=float, default=20.0, help="seconds to wait at startup before initializing board (countdown shown)")
    args = p.parse_args()

    left_pos, right_pos = _resolve_positions(args)
    print(f"Left click target:  {left_pos}")
    print(f"Right click target: {right_pos}")
    print("Press Ctrl+C to stop.")

    stream = None
    pipeline: Optional[RealtimeCCAPipeline] = None
    clicker = _MouseClicker()
    last_click_t = 0.0
    last_action = "---"

    pygame.init()
    screen = pygame.display.set_mode((args.ui_width, args.ui_height))
    pygame.display.set_caption("SSVEP Click Router Test")
    clock = pygame.time.Clock()

    f_big = pygame.font.SysFont("Arial", 72, bold=True)
    f_mid = pygame.font.SysFont("Arial", 34, bold=True)
    f_small = pygame.font.SysFont("Arial", 24)

    bw = int(args.ui_width * 0.34)
    bh = int(args.ui_height * 0.42)
    y = int(args.ui_height * 0.33)
    left_rect = pygame.Rect(int(args.ui_width * 0.10), y, bw, bh)
    right_rect = pygame.Rect(int(args.ui_width * 0.56), y, bw, bh)

    left_count = 0
    right_count = 0
    left_flash_until = 0.0
    right_flash_until = 0.0
    latest_label = "---"
    latest_peak_hz = 0.0

    try:
        # Startup pause: show a visible countdown on the UI before starting the board
        if args.startup_pause_s and args.startup_pause_s > 0:
            end_t = time.time() + float(args.startup_pause_s)
            while True:
                remaining = int(max(0, end_t - time.time()))
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt()
                    elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt()

                # draw the normal UI beneath the countdown
                _draw_ui(
                    screen,
                    f_big,
                    f_mid,
                    f_small,
                    left_rect,
                    right_rect,
                    left_count,
                    right_count,
                    args.left_hz,
                    args.right_hz,
                    latest_label,
                    latest_peak_hz,
                    left_flash_until,
                    right_flash_until,
                )

                # overlay countdown
                msg = f"Starting in {remaining} s"
                overlay = f_mid.render(msg, True, TEXT)
                screen.blit(overlay, overlay.get_rect(center=(screen.get_width() // 2, 86)))
                pygame.display.flip()

                if remaining <= 0:
                    break
                # update 4 times a second to keep UI responsive
                clock.tick(4)

        stream, fs = _init_board(args.serial_port, args.num_channels)
        print(f"Board ready at {fs:.3f} Hz")

        pipeline = RealtimeCCAPipeline(
            stream=stream,
            frequencies_hz=[args.left_hz, args.right_hz],
            sample_rate_hz=fs,
            window_s=args.window_s,
            step_s=args.step_s,
            confidence_ratio=args.confidence_ratio,
            min_absolute=args.min_score,
        )
        pipeline.start()
        print(
            f"Realtime CCA started (L={args.left_hz}Hz, R={args.right_hz}Hz, "
            f"window={args.window_s}s, step={args.step_s}s)"
        )

        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False

            label, scores, peak_hz = pipeline.get_latest()
            latest_label = label
            latest_peak_hz = peak_hz
            now = time.time()

            # stable labels are LEFT / RIGHT; map to 7.5 / 12 click targets
            action = "---"
            pos = None
            if label == "LEFT":
                action = f"{args.left_hz:.1f}Hz"
                pos = left_pos
            elif label == "RIGHT":
                action = f"{args.right_hz:.1f}Hz"
                pos = right_pos

            if pos is not None and (now - last_click_t) >= args.cooldown_s:
                if label == "LEFT":
                    left_count += 1
                    left_flash_until = now + 0.15
                elif label == "RIGHT":
                    right_count += 1
                    right_flash_until = now + 0.15

                if args.dry_run:
                    print(f"[DRY] {action} guess (peak {peak_hz:.2f} Hz) -> click {pos}")
                else:
                    if args.no_os_click:
                        print(f"{action} guess (peak {peak_hz:.2f} Hz) -> UI button clicked")
                    else:
                        clicker.click(pos[0], pos[1])
                        print(f"{action} guess (peak {peak_hz:.2f} Hz) -> clicked {pos}")
                last_click_t = now
                last_action = action
            else:
                if action != "---" and action != last_action:
                    print(f"{action} guess seen (cooldown active)")
                    last_action = action

            _draw_ui(
                screen,
                f_big,
                f_mid,
                f_small,
                left_rect,
                right_rect,
                left_count,
                right_count,
                args.left_hz,
                args.right_hz,
                latest_label,
                latest_peak_hz,
                left_flash_until,
                right_flash_until,
            )
            clock.tick(60)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pygame.quit()
        if pipeline is not None:
            pipeline.stop()
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            stream.release_session()


if __name__ == "__main__":
    main()
