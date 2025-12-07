import pygame
import math
import random
from collections import deque

pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Procedural Penrose Tiling")

BG_COLOR = (153, 102, 153)   #996699
RHOMBUS_COLOR = (153, 0, 0)  #990000

FPS = 30
phi = (1 + math.sqrt(5)) / 2

THIN = 0
THICK = 1

class Rhombus:
    def __init__(self, x, y, size, angle, rhomb_type):
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle
        self.type = rhomb_type
        self.vertices = self.compute_vertices()

    def compute_vertices(self):
        if self.type == THIN:
            alpha = math.pi / 5
        else:
            alpha = 2 * math.pi / 5
        s = self.size
        a = self.angle
        v0 = (self.x, self.y)
        v1 = (self.x + s * math.cos(a), self.y + s * math.sin(a))
        v2 = (v1[0] + s * math.cos(a + alpha), v1[1] + s * math.sin(a + alpha))
        v3 = (self.x + s * math.cos(a + alpha), self.y + s * math.sin(a + alpha))
        return [v0, v1, v2, v3]

    def draw(self, surface):
        pygame.draw.polygon(surface, RHOMBUS_COLOR, self.vertices, 2)

def generate_penrose(seed_x, seed_y, size, max_tiles=200):
    tiles = []
    frontier = deque()
    
    initial = Rhombus(seed_x, seed_y, size, 0, THICK)
    tiles.append(initial)
    for e in [(initial.vertices[i], initial.vertices[(i+1)%4]) for i in range(4)]:
        frontier.append((e, initial))

    while frontier and len(tiles) < max_tiles:
        edge, parent = frontier.popleft()
        # Randomly pick THIN or THICK
        rhomb_type = THIN if random.random() < 0.5 else THICK

        x0, y0 = edge[0]
        x1, y1 = edge[1]
        dx, dy = x1 - x0, y1 - y0
        angle = math.atan2(dy, dx)

        # Shift slightly along edge so the new rhombus doesn't overlap exactly
        shift = 0
        nx = x0 - shift * dy
        ny = y0 + shift * dx
        new_rhomb = Rhombus(nx, ny, size, angle, rhomb_type)

        # Basic collision check: skip if center is too close to existing tile
        too_close = False
        for t in tiles:
            cx = sum(v[0] for v in t.vertices) / 4
            cy = sum(v[1] for v in t.vertices) / 4
            ncx = sum(v[0] for v in new_rhomb.vertices) / 4
            ncy = sum(v[1] for v in new_rhomb.vertices) / 4
            if math.hypot(ncx - cx, ncy - cy) < size * 0.5:
                too_close = True
                break
        if too_close:
            continue

        tiles.append(new_rhomb)
        for e in [(new_rhomb.vertices[i], new_rhomb.vertices[(i+1)%4]) for i in range(4)]:
            frontier.append((e, new_rhomb))

    return tiles

def main():
    clock = pygame.time.Clock()
    running = True

    tiles = generate_penrose(WIDTH//2, HEIGHT//2, 60, max_tiles=300)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BG_COLOR)
        for t in tiles:
            t.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
