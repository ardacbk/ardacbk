"""Particles: a real Verlet simulation in a round container.

Units: one character cell is 1 wide and 2 tall, so circles are round on
screen. Position Verlet (x_new = 2x - x_old + a dt^2), 8 sub-steps of
dt = 1/60 s per frame, gravity, pairwise circle collisions resolved by
pushing overlapping particles apart, and a circular boundary constraint.

A nozzle at the upper left emits one particle per frame. They arc across,
hit the wall, slide down and pile up. Once the pile has rested, the bottom
of the container opens, the particles pour out and leave the frame, and the
wall closes again: the last frame is the empty container, like the first.
"""
import math
import sys

import numpy as np

from engine import CARD, Grid, Movie

COLS, ROWS = CARD
FPS, N = 10, 120
SUB, DT = 8, 1 / 60
G = 75.0                                  # units / s^2 (1 unit = 1 column)
CX, CY = 34.0, 16.0                       # container centre, in units
R_DRAW = 15.2                             # drawn outline radius
R_PHYS = 13.95                            # limit for particle centres (minus own radius)
NOZZLE = (26.0, 8.0)
EMIT_FIRST, EMIT_COUNT = 2, 72
OPEN_AT, OPEN_FRAMES, CLOSE_AT = 98, 6, 112
RNG_SEED = 11
STEEP = 30
YS = 0.55                                 # collision metric: y counts 0.55 so neighbours sit in adjacent cells
YW = 1.18                                 # ... but keeps off the flat floor glyphs
XS = 0.95                                 # wall metric: the pile may touch the outline sideways


# ------------------------------------------------------------- the circle
def outline(gap=0.0):
    """Circle as line art: steep parts one glyph per row, flat parts one glyph per column."""
    cells, side = {}, {}

    def open_at(x, y):
        return gap and abs(math.atan2(y - CY, x - CX) - math.pi / 2) < gap

    def tang(x, y):
        t = math.atan2(y - CY, x - CX)
        return math.degrees(math.atan2(abs(math.cos(t)), abs(math.sin(t)))), t   # 0 = horizontal tangent

    for r in range(ROWS):                                  # per-row
        dy = r * 2 + 1 - CY
        if abs(dy) >= R_DRAW:
            continue
        for sgn in (-1, 1):
            x, y = CX + sgn * math.sqrt(R_DRAW ** 2 - dy * dy), r * 2 + 1
            ang = math.degrees(math.atan2(math.sqrt(R_DRAW ** 2 - dy * dy), abs(dy)))   # tangent angle from horizontal
            if ang < STEEP or open_at(x, y):
                continue
            c = int(x // 1)
            if ang > 78:
                ch = "(" if sgn < 0 else ")"
            elif ang > 62:
                ch = "|"
            else:
                ch = "/" if (sgn < 0) == (dy < 0) else "\\"
            cells[(c, r)] = ch
            side.setdefault(sgn, {})[r] = c
    for sgn, rows in side.items():                         # bridge shallow steps with a low _
        for r in sorted(rows):
            if r + 1 in rows and abs(rows[r] - rows[r + 1]) > 1:
                lo, hi = sorted((rows[r], rows[r + 1]))
                for c in range(lo + 1, hi):
                    cells.setdefault((c, r), "_")
    for c in range(COLS):                                  # per-column
        dx = c + 0.5 - CX
        if abs(dx) >= R_DRAW:
            continue
        for sgn in (-1, 1):
            y = CY + sgn * math.sqrt(R_DRAW ** 2 - dx * dx)
            x = c + 0.5
            ang = math.degrees(math.atan2(abs(dx), math.sqrt(R_DRAW ** 2 - dx * dx)))
            if ang >= STEEP or open_at(x, y):
                continue
            r = int(y // 2)
            fy = y / 2 - r
            if fy >= 0.6:
                cells.setdefault((c, r), "_")
            elif fy >= 0.3:
                cells.setdefault((c, r), "-")
            else:
                cells.setdefault((c, r - 1), "_")
    return cells


# ------------------------------------------------------------- the physics
class World:
    def __init__(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.p = np.zeros((0, 2))
        self.o = np.zeros((0, 2))
        self.r = np.zeros(0)
        self.born = np.zeros(0, int)
        self.spawned = 0
        self.gap = 0.0

    def emit(self, f):
        big = self.spawned % 9 == 4
        r = 0.85 if big else 0.52
        v = np.array([self.rng.uniform(26, 34), self.rng.uniform(-9, 4)]) * DT
        pos = np.array(NOZZLE) + [2.5, 0]
        self.p = np.vstack([self.p, pos])
        self.o = np.vstack([self.o, pos - v])
        self.r = np.append(self.r, r)
        self.born = np.append(self.born, f)
        self.spawned += 1

    def substep(self):
        p, o = self.p, self.o
        if len(p) == 0:
            return
        v = p - o
        sp = np.hypot(v[:, 0], v[:, 1])
        damp = np.where(sp < 0.02, 0.85, 0.9985)[:, None]
        v = v * damp
        new = p + v + np.array([0.0, G * DT * DT])
        o[:] = p
        p[:] = new
        for _ in range(4):
            d = (p[:, None, :] - p[None, :, :]) * [1.0, YS]
            dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(len(p))
            ov = (self.r[:, None] + self.r[None, :]) - dist
            np.fill_diagonal(ov, 0)
            m = ov > 0
            if m.any():
                push = np.where(m, 0.5 * ov / dist, 0)[..., None] * d
                p += 0.8 * push.sum(1) / [1.0, YS]
            self.wall()

    def wall(self):
        p, o = self.p, self.o
        d = (p - [CX, CY]) * [XS, YW]
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        lim = R_PHYS - 0.3 * self.r
        ang = np.arctan2(d[:, 1], d[:, 0])
        in_gap = (np.abs(ang - math.pi / 2) < self.gap) if self.gap else np.zeros(len(p), bool)
        hit = (dist > lim) & ~in_gap
        if hit.any():
            n = d[hit] / dist[hit, None]
            tgt = np.array([CX, CY]) + n * lim[hit, None] / [XS, YW]
            vel = p[hit] - o[hit]
            vn = (vel * n).sum(1)
            vel = vel - (1.0 + 0.45) * np.where(vn > 0, vn, 0)[:, None] * n
            vt = vel - (vel * n).sum(1)[:, None] * n
            vel = vel - 0.04 * vt                        # wall friction
            p[hit] = tgt
            o[hit] = tgt - vel

    def frame(self):
        for _ in range(SUB):
            self.substep()
        keep = self.p[:, 1] < 34.0                       # fell out of the card
        if not keep.all():
            self.p, self.o, self.r, self.born = self.p[keep], self.o[keep], self.r[keep], self.born[keep]
            self.disp = [x for x, k in zip(self.disp, keep) if k] if hasattr(self, "disp") else []


# ------------------------------------------------------------- drawing
def build():
    w = World()
    w.disp = []
    frames = []
    base = Grid(COLS, ROWS)
    ring = outline()
    for (c, r), ch in ring.items():
        base.put(c, r, ch, "t2")
    nz_c, nz_r = int(NOZZLE[0]) - 1, int(NOZZLE[1] // 2)
    for f in range(N):
        # events
        if EMIT_FIRST <= f < EMIT_FIRST + EMIT_COUNT:
            w.emit(f)
            w.disp.append(None)
        w.gap = 0.0
        if f >= OPEN_AT:
            w.gap = math.radians(34) * min((f - OPEN_AT + 1) / OPEN_FRAMES, 1.0)
        closing = f >= CLOSE_AT
        if closing:
            w.gap = math.radians(34) * max(1 - (f - CLOSE_AT + 1) / 4, 0.0) if len(w.p) == 0 else w.gap
        w.frame()

        g = Grid(COLS, ROWS)
        cur = outline(w.gap) if w.gap else ring
        for (c, r), ch in cur.items():
            g.put(c, r, ch, "t2")
        if f < OPEN_AT or True:
            g.put(nz_c, nz_r, "[", "t2")
            g.put(nz_c + 1, nz_r, "=", "t2")

        cells = {}
        for i in range(len(w.p)):
            x, y = w.p[i]
            v = w.p[i] - w.o[i]
            spf = math.hypot(*v) * SUB                     # units per frame
            c, r = int(x // 1), int(y // 2)
            d = w.disp[i]
            if d is not None and spf < 1.5:                # hysteresis: settled things stay put
                dc, dr = d
                if abs(x - (dc + 0.5)) < 0.95 and abs(y - (dr * 2 + 1)) < 1.6:
                    c, r = dc, dr
            w.disp[i] = (c, r)
            if r >= ROWS - 1 or not (0 <= c < COLS):
                continue
            age = f - w.born[i]
            if age < 2:
                cls, ch = "a", "o"
            else:
                cls = "t3" if spf < 1.0 else ("t4" if spf < 3.0 else "t5")
                ch = "O" if w.r[i] > 1.0 else "o"
            if age < 2:
                score = 1e3
            else:
                score = spf
            if (c, r) not in cells or cells[(c, r)][0] < score:
                cells[(c, r)] = (score, ch, cls)
        for (c, r), (_, ch, cls) in cells.items():
            g.put(c, r, ch, cls)
        n = len(w.p)
        for i, line in enumerate(("verlet", f"n = {n:03d}", f"dt 1/60 x {SUB}")):
            g.text(2, 7 + i, line, "t2")
        frames.append(g)
    return frames


if __name__ == "__main__":
    frames = build()
    m = Movie(COLS, ROWS, title="particle-physics-simulation",
              desc="An ASCII Verlet simulation. A nozzle at the upper left of a round container fires a stream of "
                   "particles that arc across, bounce off the wall, slide down and pile up at the bottom. The "
                   "newest particle is drawn in red. Then the bottom opens, the pile pours out and the "
                   "empty container is ready for the next run.")
    m.add(frames, FPS)
    if "--preview" in sys.argv:
        m.preview("particles.png", times=(0.5, 2.0, 3.5, 5.0, 7.0, 9.0, 10.5, 11.9))
        m.preview("particles-light.png", times=(3.5, 8.0), theme="light")
    m.write("particles.svg")
