"""sand-simulation: the falling-sand automaton itself, drawn one cell per character.

Every cell holds an element (empty, stone, sand, water). Each tick the grid is
swept bottom-up, each row in a random left/right order (seeded), and every
element runs its own update():
  sand   falls, slides diagonally, sinks through water (slowly: water is thick)
  water  falls, slides diagonally, then flows sideways
  stone  stays put
A "mouse" pours sand (left click) or water (right click) at the cursor.

Loop: empty scene -> water -> sand -> more water until it overflows -> the
sources stop, the floor and the roof of the ledge open, everything drains off
the bottom -> they close again on an empty scene.

The card uses the fine grid, so the automaton has about twice the cells and
its piles settle into readable slopes. Water is shaded by depth: a bright
rippling surface over a body that darkens as it deepens.
"""
import sys

import numpy as np

from engine import CARD_FINE, FINE, Grid, Movie

COLS, ROWS = CARD_FINE
FPS, N = 10, 110
TICKS = 2                        # automaton ticks per frame

EMPTY, STONE, SAND, WATER = 0, 1, 2, 3
H = 22                           # simulated rows 0..21 (row 21 is the floor)
FL = H - 1
XMIN, XMAX = 2, 92               # left edge is a wall, the right edge is a drop

# ---------------------------------------------------------------- the level
ROOF = (10, 11)                  # two rows of stone over a hollow ledge
LEDGE = (2, 41)                  # columns of the ledge block
BASIN = (42, 72)                 # basin interior
WALL = (73, 74)                  # right wall of the basin
WALL_TOP = 14
SHELF = (59, 68, 7)              # small floating stone shelf (c0, c1, row)
SAND_COLS = (38, 31)
WATER_COL = 64


def level():
    m = np.zeros((H, COLS), np.int8)
    m[FL, XMIN:XMAX + 1] = STONE                                   # floor
    m[ROOF[0]:ROOF[1] + 1, LEDGE[0]:LEDGE[1] + 1] = STONE          # roof of the ledge
    m[ROOF[1] + 1:FL, LEDGE[0]:LEDGE[0] + 2] = STONE               # left pillar
    m[ROOF[1] + 1:FL, LEDGE[1] - 1:LEDGE[1] + 1] = STONE           # right pillar
    m[WALL_TOP:FL, WALL[0]:WALL[1] + 1] = STONE                    # basin wall
    m[SHELF[2], SHELF[0]:SHELF[1] + 1] = STONE
    return m


# trapdoors: stone cells that open for the drain (cell -> frame it opens)
rng_doors = np.random.default_rng(3)
DOORS = {}
for c in range(6, LEDGE[1] - 1):
    for r in ROOF:
        DOORS[(r, c)] = int(rng_doors.integers(83, 88))
for c in list(range(4, LEDGE[1] - 1)) + list(range(BASIN[0], BASIN[1] + 1)) + list(range(WALL[1] + 1, XMAX + 1)):
    DOORS[(FL, c)] = int(rng_doors.integers(83, 89))
CLOSE_AT = 108

# ---------------------------------------------------------------- cursor timeline
# (frame, column, tool)
STOPS = [(3, WATER_COL, "water"), (28, WATER_COL, "water"), (36, SAND_COLS[0], "sand"),
         (46, SAND_COLS[0], "sand"), (51, SAND_COLS[1], "sand"), (58, SAND_COLS[1], "sand"),
         (64, WATER_COL, "water"), (81, WATER_COL, "water")]
POUR = {"water": [(4, 28), (64, 80)], "sand": [(37, 58)]}


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def cursor(f):
    """-> (column, tool, pouring) or None"""
    if f < STOPS[0][0] or f > STOPS[-1][0]:
        return None
    for (f0, c0, t0), (f1, c1, t1) in zip(STOPS, STOPS[1:]):
        if f0 <= f <= f1:
            x = c0 + (c1 - c0) * ease((f - f0) / max(f1 - f0, 1))
            tool = t1 if f > f0 and c1 != c0 else t0
            pouring = any(a <= f <= b for a, b in POUR[tool]) and (c0 == c1)
            return int(round(x)), tool, pouring
    return None


# ---------------------------------------------------------------- the automaton
class World:
    def __init__(self, seed=11):
        self.m = level()
        self.rng = np.random.default_rng(seed)
        self.moved = np.zeros_like(self.m, bool)      # moved during this frame
        self.fell = np.zeros_like(self.m, bool)

    def at(self, r, c):
        """element at (r, c); the right edge and the bottom are drops (-1)."""
        if r >= H or c > XMAX:
            return -1
        if c < XMIN or r < 0:
            return STONE
        return int(self.m[r, c])

    def put(self, r, c, e, fell=False):
        if r >= H or c > XMAX:
            return
        self.m[r, c] = e
        self.moved[r, c] = True
        self.fell[r, c] = fell

    def tick(self, pour):
        m, rng = self.m, self.rng
        tick_moved = np.zeros_like(m, bool)
        for r in range(H - 1, -1, -1):
            order = range(XMIN, XMAX + 1) if rng.random() < 0.5 else range(XMAX, XMIN - 1, -1)
            for c in order:
                e = m[r, c]
                if e not in (SAND, WATER) or tick_moved[r, c]:
                    continue
                dirs = (-1, 1) if rng.random() < 0.5 else (1, -1)
                b = self.at(r + 1, c)
                if b in (EMPTY, -1):
                    self._move(r, c, r + 1, c, tick_moved, True)
                    continue
                if e == SAND:
                    if b == WATER and rng.random() < 0.45:
                        self._swap(r, c, r + 1, c, tick_moved)
                        continue
                    for d in dirs:
                        t = self.at(r + 1, c + d)
                        if t in (EMPTY, -1) and self.at(r, c + d) in (EMPTY, WATER, -1):
                            self._move(r, c, r + 1, c + d, tick_moved, True)
                            break
                    continue
                for d in dirs:                       # water: diagonal, then sideways
                    if self.at(r + 1, c + d) in (EMPTY, -1) and self.at(r, c + d) in (EMPTY, -1):
                        self._move(r, c, r + 1, c + d, tick_moved, True)
                        break
                else:
                    for d in dirs:
                        if self.at(r, c + d) in (EMPTY, -1):
                            self._move(r, c, r, c + d, tick_moved, False)
                            break
        # the mouse
        for tool, col in pour:
            if tool == "water":
                n = 6 + int(rng.random() < 0.7)
            else:
                n = 6 + int(rng.random() < 0.6)
            for _ in range(n):
                cc = col + int(rng.integers(-2, 3))
                if m[1, cc] == EMPTY:
                    self.put(1, cc, WATER if tool == "water" else SAND, True)
                    tick_moved[1, cc] = True

    def _move(self, r, c, r2, c2, tick_moved, fell):
        e = self.m[r, c]
        self.m[r, c] = EMPTY
        self.moved[r, c] = False
        self.fell[r, c] = False
        if r2 < H and c2 <= XMAX:
            self.put(r2, c2, e, fell)
            tick_moved[r2, c2] = True

    def _swap(self, r, c, r2, c2, tick_moved):
        a, b = self.m[r, c], self.m[r2, c2]
        self.m[r, c], self.m[r2, c2] = b, a
        for rr, cc in ((r, c), (r2, c2)):
            self.moved[rr, cc] = True
            self.fell[rr, cc] = False
            tick_moved[rr, cc] = True


# ---------------------------------------------------------------- drawing
LEGEND_ROW = ROWS - 1


def stone_class(m, r, c):
    """t3 where a stone cell touches the open air, t2 inside the block."""
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and XMIN <= cc <= XMAX and m[rr, cc] != STONE:
            return "t3"
    return "t2"


def draw(world, stone, f):
    g = Grid(COLS, ROWS)
    m = world.m
    for r in range(H):
        for c in range(XMIN, XMAX + 1):
            e = m[r, c]
            if stone[r, c]:
                g.put(c, r, "#", stone_class(stone, r, c))
            elif e == SAND:
                ch = ".:;:"[(c * 7 + r * 3) % 4]
                g.put(c, r, ch, "a" if world.moved[r, c] else "b")
            elif e == WATER:
                air = r == 0 or m[r - 1, c] == EMPTY
                below = world.at(r + 1, c)
                depth = 0                          # water cells above this one
                while r - depth - 1 >= 0 and m[r - depth - 1, c] == WATER:
                    depth += 1
                if (below in (EMPTY, -1) or world.fell[r, c]) and r <= FL - 3:
                    g.put(c, r, "|", "t4")         # falling
                elif air and below not in (EMPTY, -1):
                    g.put(c, r, "-" if ((c // 3) + f // 8) % 3 == 0 else "~", "t5")   # the rippling surface
                elif depth < 3:
                    g.put(c, r, "~", "t4")
                elif depth < 6:
                    g.put(c, r, "~" if (c + r) % 2 else "-", "t3")
                else:
                    g.put(c, r, "-" if (c + r) % 3 else ".", "t2")
    cur = cursor(f)
    if cur:
        col, tool, pouring = cur
        g.put(col, 0, "v" if pouring else "+", "t4")
        g.text(col + 2, 0, tool, "t3")
    # legend
    g.put(2, LEGEND_ROW, "#", "t2")
    g.text(4, LEGEND_ROW, "stone", "t2")
    g.put(11, LEGEND_ROW, ":", "b")
    g.text(13, LEGEND_ROW, "sand", "t2")
    g.put(19, LEGEND_ROW, "~", "t4")
    g.text(21, LEGEND_ROW, "water", "t2")
    return g


def build():
    w = World()
    stone = np.zeros((H, COLS), bool)
    frames, counts = [], []
    for f in range(N):
        # trapdoors
        base = level()
        for (r, c), t in DOORS.items():
            if t <= f < CLOSE_AT:
                base[r, c] = EMPTY
        # stone cells are static scenery layered under the moving elements
        was = w.m == STONE
        now = base == STONE
        w.m[was & ~now] = EMPTY
        w.m[~was & now & (w.m == EMPTY)] = STONE
        if f == CLOSE_AT:                            # the scene must already be empty
            left = int(((w.m == SAND) | (w.m == WATER)).sum())
            print("cells still falling when the doors close:", left)
            w.m[(w.m == SAND) | (w.m == WATER)] = EMPTY
        stone = w.m == STONE
        w.moved[:] = False
        for k in range(TICKS):
            pour = []
            cur = cursor(f)
            if cur and cur[2]:
                pour.append((cur[1], cur[0]))
            w.tick(pour)
        stone = w.m == STONE
        frames.append(draw(w, stone, f))
        counts.append(int((w.m == SAND).sum()) * 1000 + int((w.m == WATER).sum()))
    return frames, counts


def main():
    frames, counts = build()
    print("sand/water cells at selected frames:",
          {i: (counts[i] // 1000, counts[i] % 1000) for i in (0, 20, 40, 60, 78, 90, N - 1)})
    left = counts[-1]
    print("left at end:", left)
    m = Movie(COLS, ROWS, pad=(12, 12), cell=FINE, title="sand-simulation",
              desc="A falling-sand cellular automaton drawn one cell per character. Water pours into a stone "
                   "basin, sand pours onto a ledge, piles up, slides over the edge and sinks through the water "
                   "until the basin overflows; then the floor opens, everything drains away and the loop restarts.")
    m.add(frames, FPS)
    if "--preview" in sys.argv:
        m.preview("sand.png", times=(0.0, 2.0, 3.8, 5.6, 7.6, 8.8))
        m.preview("sand-light.png", times=(6.2, 7.8), theme="light")
        m.preview("sand-seam.png", times=(10.8, 0.0))
    m.write("sand.svg")


if __name__ == "__main__":
    main()
