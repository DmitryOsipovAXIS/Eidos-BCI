import pygame
import sys

pygame.init()

# Display setup
info = pygame.display.get_desktop_sizes()
width, height = info[0]
screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
pygame.display.set_caption("SSVEP Stimulus Test")
clock = pygame.time.Clock()

# Frequencies
freq_left = 10  # Hz
freq_right = 15  # Hz

# Stimulus parameters
square_size = 200
gap = 100
center_x = width // 2
center_y = height // 2

left_square = pygame.Rect(center_x - gap - square_size, center_y - square_size // 2, square_size, square_size)
right_square = pygame.Rect(center_x + gap, center_y - square_size // 2, square_size, square_size)

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (50, 50, 50)

# State tracking
left_period = 1 / freq_left
right_period = 1 / freq_right

# Font for info
font = pygame.font.Font(None, 36)

running = True
start_time = pygame.time.get_ticks() / 1000

while running:
    dt = pygame.time.get_ticks() / 1000 - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Determine which squares to draw
    left_time = dt % left_period
    right_time = dt % right_period

    left_on = left_time < (left_period / 2)
    right_on = right_time < (right_period / 2)

    # Draw background
    screen.fill(gray)

    # Draw left square (10Hz)
    if left_on:
        pygame.draw.rect(screen, white, left_square)
    else:
        pygame.draw.rect(screen, black, left_square)

    # Draw right square (15Hz)
    if right_on:
        pygame.draw.rect(screen, white, right_square)
    else:
        pygame.draw.rect(screen, black, right_square)

    # Draw info text
    info_text = font.render(f"Left: {freq_left}Hz | Right: {freq_right}Hz | Press ESC to quit", True, white)
    screen.blit(info_text, (20, 20))

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
sys.exit()
