"""Pygame screen runners for SSVEP stimuli."""
from __future__ import annotations

import sys
from typing import Optional

from pipeline.realtime_pipeline import is_confident

import numpy as np
import pygame
from datetime import datetime

from acquisition.recording import RecordingResult, start_recording_thread
from pipeline.realtime_pipeline import RealtimeCCAPipeline
from ui.drawing import (
    BG,
    BLUE,
    DIM,
    GREY,
    RED,
    WHITE,
    draw_dual,
    draw_fps,
    draw_inference_box,
    draw_single,
    draw_text_center,
)

import asyncio
import json

SINGLE_BOX_W_RATIO = 0.35
SINGLE_BOX_H_RATIO = 0.50
SINGLE_BOX_LEFT_X_RATIO = 0.12
SINGLE_BOX_RIGHT_X_RATIO = 0.53

DUAL_BOX_W_RATIO = 0.32
DUAL_BOX_H_RATIO = 0.46
DUAL_LEFT_X_RATIO = 0.08
DUAL_RIGHT_X_RATIO = 0.60

REC_THREAD_EXTRA_TIMEOUT_S = 5


def _set_mode(size: tuple[int, int], fullscreen: bool) -> pygame.Surface:
    flags = pygame.SCALED
    if fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        return pygame.display.set_mode(size, flags, vsync=0)
    except TypeError:
        return pygame.display.set_mode(size, flags)


def _screen_size(fullscreen: bool) -> tuple[int, int]:
    info = pygame.display.Info()
    if fullscreen:
        return info.current_w, info.current_h
    return int(info.current_w * 0.9), int(info.current_h * 0.9)


def _make_fonts(H: int) -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))
    fsml = pygame.font.SysFont("Arial", max(18, H // 36))
    return fbig, fmed, fsml


def start_menu(fullscreen: bool) -> dict:
    pygame.init()
    W, H = _screen_size(fullscreen)
    screen = _set_mode((W, H), fullscreen)
    pygame.display.set_caption("SSVEP Start Menu")
    clock = pygame.time.Clock()
    fbig, fmed, fsml = _make_fonts(H)

    options = [
        ("1", "Left 7.5 Hz for 2 minutes",
         {"mode": "single", "side": "left", "hz": 7.5, "duration": 120.0}),
        ("2", "Right 12 Hz for 2 minutes",
         {"mode": "single", "side": "right", "hz": 12.0, "duration": 120.0}),
        ("3", "Alternate 7.5/12 Hz for 2 minutes",
         {"mode": "alt", "left_hz": 7.5, "right_hz": 12.0, "alt_total": 120.0, "alt_block": 30.0}),
        ("4", "Alternate BOTH flickers (7.5/12) for 2 minutes",
         {"mode": "alt", "left_hz": 7.5, "right_hz": 12.0, "alt_total": 120.0, "alt_block": 30.0, "both_flicker": True}),
        ("5", "Use CLI settings (single or alt)",
         {"mode": "cli"}),
        ("6", "Live mode (no timer, no save)", {"mode": "live"}),
        ("7", "One-box live 12 Hz (no timer, no save)", {"mode": "one_box_live"}),
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
        draw_text_center(screen, fbig, "SSVEP START", H // 6, GREY)

        y = H // 3
        option_rects = []
        for k, text, payload in options:
            lbl = f"{k}. {text}"
            surf = fmed.render(lbl, True, GREY)
            rect = surf.get_rect(center=(W // 2, y))
            screen.blit(surf, rect)
            option_rects.append((rect.inflate(40, 16), payload))
            y += 46

        draw_text_center(
            screen, fsml, "Press 1-5 or click an option", H - 90, GREY)
        draw_text_center(screen, fsml, "ESC to quit", H - 60, GREY)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return selected


def run_single(
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
    W, H = _screen_size(args.fullscreen)
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP Single Target")
    clock = pygame.time.Clock()
    fbig, fmed, fsml = _make_fonts(H)

    bw = int(W * SINGLE_BOX_W_RATIO)
    bh = int(H * SINGLE_BOX_H_RATIO)
    cy = H // 2
    x_ratio = SINGLE_BOX_LEFT_X_RATIO if side == "left" else SINGLE_BOX_RIGHT_X_RATIO
    rect = pygame.Rect(int(W * x_ratio), cy - bh // 2, bw, bh)

    rec_thread, start_event, result = None, None, None
    if stream is not None:
        rec_thread, start_event, result = start_recording_thread(
            stream, label, duration_s, args.discard
        )

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
        draw_single(screen, rect, phase < 0.5, label,
                    hz, fbig, fmed, clock.get_fps())

        if rt_pipeline is not None:
            rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
            draw_inference_box(screen, fmed, fsml, rt_label,
                               rt_scores, rt_peak_hz)

        pygame.display.flip()

        if stream is not None and not started:
            start_event.set()
            started = True
        clock.tick(args.monitor_hz)

    pygame.quit()

    if stream is not None and rec_thread is not None:
        rec_thread.join(timeout=duration_s + REC_THREAD_EXTRA_TIMEOUT_S)

    if result is not None and result.error:
        print(f"Recording error: {result.error}", flush=True)
        return None

    return result.eeg if result is not None else None


def run_alternating(
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
    W, H = _screen_size(args.fullscreen)
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP Alternating")
    clock = pygame.time.Clock()
    fbig, fmed, fsml = _make_fonts(H)

    bw = int(W * DUAL_BOX_W_RATIO)
    bh = int(H * DUAL_BOX_H_RATIO)
    cy = H // 2
    lrect = pygame.Rect(int(W * DUAL_LEFT_X_RATIO), cy - bh // 2, bw, bh)
    rrect = pygame.Rect(int(W * DUAL_RIGHT_X_RATIO), cy - bh // 2, bw, bh)

    left_step = left_hz / args.monitor_hz
    right_step = right_hz / args.monitor_hz
    blocks = max(1, int(alt_total_s / alt_block_s))
    labels = [(label_start + i) % 2 for i in range(blocks)]
    results: list[np.ndarray] = []
    left_phase = right_phase = 0.0

    for idx, label in enumerate(labels):
        rec_thread, start_event, result = None, None, None
        if stream is not None:
            rec_thread, start_event, result = start_recording_thread(
                stream, label, alt_block_s, args.discard
            )

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

            draw_dual(
                screen, lrect, rrect,
                left_on, right_on,
                label, left_hz, right_hz,
                fbig, fmed, clock.get_fps(), block_left,
            )

            if rt_pipeline is not None:
                rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
                draw_inference_box(screen, fmed, fsml,
                                   rt_label, rt_scores, rt_peak_hz)

            pygame.display.flip()

            if stream is not None and not started:
                start_event.set()
                started = True
            clock.tick(args.monitor_hz)

        if rec_thread is not None:
            rec_thread.join(timeout=alt_block_s + REC_THREAD_EXTRA_TIMEOUT_S)

        if result is not None and result.error:
            print(f"Block {idx + 1} error: {result.error}", flush=True)
        elif result is not None and result.eeg is not None:
            results.append(result.eeg)

        if not running:
            break
    pygame.quit()
    return results

def run_live(
        args, left_hz: float, right_hz: float, rt_pipeline: Optional[RealtimeCCAPipeline], broadcast=None, loop=None,
) -> None:
    pygame.init()
    W, H = _screen_size(args.fullscreen)
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP LIVE")
    clock = pygame.time.Clock()
    fbig, fmed, fsml = _make_fonts(H)

    bw = int(W * DUAL_BOX_W_RATIO)
    bh = int(H * DUAL_BOX_H_RATIO)
    cy = H // 2
    
    lrect = pygame.Rect(int(W*DUAL_LEFT_X_RATIO), cy-bh // 2,bw,bh)
    rrect = pygame.Rect(int(W*DUAL_RIGHT_X_RATIO), cy-bh//2,bw,bh)

    left_step = left_hz / args.monitor_hz
    right_step = right_hz / args.monitor_hz
    left_phase = right_phase = 0.0
    left_count = right_count = neutral_count = 0
    last_label = ""
    last_timestamp = 0
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        left_phase = (left_phase + left_step) % 1.0
        right_phase = (right_phase + right_step) % 1.0

        if rt_pipeline is not None:
            rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
            ts = datetime.now().timestamp()
            if (rt_label != last_label or ts - last_timestamp > 5) and rt_pipeline.is_confident(rt_scores):
                if rt_label == "LEFT":
                    left_count += 1
                elif rt_label == "RIGHT":
                    right_count += 1
                elif rt_label == "NEUTRAL":
                    neutral_count += 1
                action_map = {"RIGHT": "WIDGET", "LEFT": "FUNCTION"}
                action = action_map.get(rt_label, rt_label)
                if broadcast is not None and loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast(json.dumps({"side": rt_label, "type": action, "scores": rt_scores, "timestamp": ts, "last_timestamp": last_timestamp, "time_diff": ts-last_timestamp})), loop
                    )
                last_timestamp = ts
            last_label = rt_label
        else:
            rt_label, rt_scores, rt_peak_hz = "---", [], 0.0
            last_timestamp = 0

        screen.fill(BG)
        pygame.draw.rect(screen, WHITE if left_phase < 0.5 else DIM, lrect)
        pygame.draw.rect(screen, WHITE if right_phase < 0.5 else DIM, rrect)
        pygame.draw.rect(screen, BLUE, lrect, 4)
        pygame.draw.rect(screen, RED, rrect, 4)
        for text, color, x, y in [
            (str(left_count), BLUE, lrect.centerx, lrect.top - 50),
            (str(neutral_count), GREY, W // 2, H // 8),
            (str(right_count), RED, rrect.centerx, rrect.top - 50),
        ]:
            s = fbig.render(text, True, color)
            screen.blit(s, s.get_rect(center=(x, y)))
        draw_text_center(screen, fmed, f"Left {left_hz:.1f} Hz", lrect.bottom + 22, BLUE)
        draw_text_center(screen, fmed, f"Right {right_hz:.1f} Hz", rrect.bottom + 22, RED)
        draw_fps(screen, fmed, clock.get_fps())
        draw_inference_box(screen, fmed, fsml, rt_label, rt_scores, rt_peak_hz)

        pygame.display.flip()
        clock.tick(args.monitor_hz)

    pygame.quit()

def one_box_live(args, right_hz: float, rt_pipeline: Optional[RealtimeCCAPipeline],broadcast=None, loop=None) -> None:
    pygame.init()
    W, H = _screen_size(args.fullscreen)
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP LIVE 12 HERTZ")
    clock = pygame.time.Clock()
    fbig, fmed, fsml = _make_fonts(H)

    bw = int(W * DUAL_BOX_W_RATIO)
    bh = int(H * DUAL_BOX_H_RATIO)

    rect = pygame.Rect((W - bw) // 2, (H - bh) // 2, bw, bh)
    right_step = right_hz / args.monitor_hz
    right_phase = 0.0
    right_count = 0
    neutral_count = 0
    last_label = ""
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        right_phase = (right_phase + right_step) % 1.0

        if rt_pipeline is not None:
            rt_label, rt_scores, rt_peak_hz = rt_pipeline.get_latest()
            if rt_label != last_label:
                if rt_label == "RIGHT":
                    right_count += 1
                elif rt_label == ("NEUTRAL",):
                    neutral_count += 1
                action_map = {"RIGHT": "WIDGET", "LEFT": "FUNCTION"}
                action = action_map.get(rt_label, rt_label)
                if broadcast is not None and loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast(json.dumps({"side": rt_label, "type": action, "scores": rt_scores})), loop
                    )
            last_label = rt_label
        else:
            rt_label, rt_scores, rt_peak_hz = "---", [], 0.0

        screen.fill(BG)
        pygame.draw.rect(screen, WHITE if right_phase < 0.5 else DIM, rect)
        pygame.draw.rect(screen, RED, rect, 4)

        for text, color, x, y in [
            (str(right_count), RED, rect.centerx, rect.top - 50),
            (str(neutral_count), GREY, W // 2, H // 8),
        ]:
            s = fbig.render(text, True, color)
            screen.blit(s, s.get_rect(center=(x, y)))

        draw_text_center(screen, fmed, f"{right_hz:.1f} Hz", rect.bottom + 22, RED)
        draw_fps(screen, fmed, clock.get_fps())
        draw_inference_box(screen, fmed, fsml, rt_label, rt_scores, rt_peak_hz)

        pygame.display.flip()
        clock.tick(args.monitor_hz)

    pygame.quit()
