"""Pygame drawing helpers for SSVEP stimuli and UI overlays."""
from __future__ import annotations

import pygame

BG = (0, 0, 0)
WHITE = (255, 255, 255)
DIM = (20, 20, 20)
GREY = (180, 180, 180)
BLUE = (80, 120, 255)
RED = (255, 80, 80)


def draw_text_center(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    y: int,
    color: tuple,
) -> None:
    surf = font.render(text, True, color)
    screen.blit(surf, surf.get_rect(center=(screen.get_width() // 2, y)))


def draw_fps(screen: pygame.Surface, font: pygame.font.Font, fps: float) -> None:
    surf = font.render(f"FPS {fps:0.1f}", True, GREY)
    screen.blit(surf, surf.get_rect(center=(screen.get_width() - 90, 26)))


def draw_inference_box(
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
    pygame.draw.rect(screen, (90, 90, 90),
                     (bx, by, bw, bh), 2, border_radius=8)

    col = BLUE if label == "LEFT" else (RED if label == "RIGHT" else GREY)
    s = fmed.render(label, True, col)
    screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 24)))

    hz_text = f"{peak_hz:.1f} Hz" if peak_hz > 0 else "---"
    s = fsml.render(hz_text, True, GREY)
    screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 52)))

    if scores and len(scores) >= 2:
        parts = f"L:{scores[0]:.3f}  R:{scores[1]:.3f}"
        s = fsml.render(parts, True, (110, 110, 110))
        screen.blit(s, s.get_rect(center=(bx + bw // 2, by + 74)))


def draw_single(
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
    draw_text_center(screen, fbig, "FOCUS", rect.top - 40, col)
    draw_text_center(screen, fmed, f"{hz:0.1f} Hz", rect.bottom + 22, GREY)
    draw_fps(screen, fmed, fps)


def draw_dual(
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
    draw_text_center(screen, fbig, "LOOK LEFT" if label ==
                     0 else "LOOK RIGHT", 60, BLUE if label == 0 else RED)
    draw_text_center(
        screen, fmed, f"Left {left_hz:0.1f} Hz", lrect.bottom + 22, BLUE)
    draw_text_center(
        screen, fmed, f"Right {right_hz:0.1f} Hz", rrect.bottom + 22, RED)
    draw_text_center(screen, fmed, f"Block {block_left_s:0.1f}s", 100, GREY)
    draw_fps(screen, fmed, fps)
