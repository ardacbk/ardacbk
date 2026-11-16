"""Alignment: a wireframe plane learning to align with a fixed target plane.

A tiny real environment is simulated and a linear policy is trained on it with
evolution strategies (seeded). The state is the plane's tilt about two axes
plus a height offset relative to the fixed target plane, the action is a small
bounded correction to each, the reward is the negative misalignment (plus a
small action penalty) and an episode lasts a fixed number of steps.

Snapshots of the policy are taken early, in the middle and at the end of
training, together with the learning curve. The loop replays one rollout per
snapshot from the same misaligned start pose. Both planes are drawn in a fixed
oblique projection (a level plane is a parallelogram); misalignment shows up as
corner heights (two tilts + a height offset), joined to the target corners by
short error posts. Lines are rasterised one glyph per column or per row.
"""
import math
import sys

import numpy as np

from engine import CARD, Grid, Movie

COLS, ROWS = CARD
FPS = 10

# ---------------------------------------------------------------- environment
T = 32                                   # steps per episode
AMAX = 0.12                              # bound of each correction per step
NOISE = 0.012                            # process noise per step
START = np.array([0.34, -0.4, 0.5])      # tilt a, tilt b, height offset
TOL = 0.08                               # aligned when |state| stays below this
ACT_COST = 0.5
INIT = 1.0
MIX = np.eye(3) + 0.3 * np.array([[0, 1, 0], [-1, 0, 1], [0, 1, 0]])   # actuators are cross-coupled
MID = 18                                 # training iteration of the middle snapshot


def misalignment(s):
    return float(np.linalg.norm(s))


def rollout(W, seed, start=START, trace=False):
    rng = np.random.default_rng(seed)
    s = np.array(start, float)
    a_prev = np.zeros(3)                 # one step of actuator delay
    ret, states = 0.0, [s.copy()]
    for _ in range(T):
        a = np.clip(W @ s, -AMAX, AMAX)
        s = s + MIX @ a_prev + NOISE * rng.standard_normal(3)
        a_prev = a
        ret -= misalignment(s) + ACT_COST * float(a @ a) * 10
        states.append(s.copy())
    return (ret, np.array(states)) if trace else ret


def train(iters=60, pairs=5, sigma=0.06, lr=0.02, seed=3):
    rng = np.random.default_rng(seed)
    W = -INIT * np.eye(3) + 0.12 * rng.standard_normal((3, 3))
    eval_starts = [START + 0.12 * np.random.default_rng(100 + i).standard_normal(3) for i in range(8)]

    def score(Wc):
        return float(np.mean([rollout(Wc, 500 + i, st) for i, st in enumerate(eval_starts)]))

    snaps, curve = [], []
    for it in range(iters + 1):
        curve.append(score(W))
        snaps.append(W.copy())
        if it == iters:
            break
        eps = rng.standard_normal((pairs, 3, 3))
        start = START + 0.12 * rng.standard_normal(3)
        sd = int(rng.integers(1 << 30))
        rp = np.array([rollout(W + sigma * e, sd, start) for e in eps])
        rm = np.array([rollout(W - sigma * e, sd, start) for e in eps])
        scale = np.concatenate([rp, rm]).std() + 1e-9
        W = W + lr / (pairs * sigma) * ((rp - rm)[:, None, None] * eps).sum(0) * sigma / scale
    return snaps, np.array(curve)


# ---------------------------------------------------------------- line art
def flat_glyph(y):
    """Row and glyph for a flat line at vertical position y (in rows; a cell spans [r, r+1))."""
    r = math.floor(y)
    f = y - r
    if f < 0.25:
        return r - 1, "_"
    if f < 0.62:
        return r, "-"
    return r, "_"


def emit(cells, items, dash):
    for k, (key, g) in enumerate(items):
        if dash and k % 2:
            continue
        cells[key] = g


def col_line(cells, p0, p1, dash=False):
    """One glyph per column (for lines that run mostly sideways). Points are (x, y) in cells/rows."""
    (x0, y0), (x1, y1) = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
    dy = (y1 - y0) / (x1 - x0) if x1 > x0 else 0.0
    items = []
    for c in range(math.floor(x0), math.floor(x1) + 1):
        x = min(max(c + 0.5, x0), x1)
        y = y0 + dy * (x - x0)
        if abs(dy) > 0.5:
            r, g = math.floor(y), "/" if dy < 0 else "\\"
        else:
            r, g = flat_glyph(y)
        items.append(((r, c), g))
    emit(cells, items, dash)


def row_line(cells, p0, p1, dash=False):
    """One glyph per row (for lines that run mostly up the page); p0 is the start."""
    (x0, y0), (x1, y1) = p0, p1
    lo, hi = min(y0, y1), max(y0, y1)
    rows = [r for r in range(math.floor(lo), math.floor(hi) + 1) if lo <= r + 0.5 <= hi] or [math.floor((lo + hi) / 2)]
    rows.sort(key=lambda r: abs(r + 0.5 - y0))
    per_row = abs(x1 - x0) / max(abs(y1 - y0), 1e-6)       # columns per row
    items = []
    for r in rows:
        t = (r + 0.5 - y0) / (y1 - y0) if y1 != y0 else 0
        x = x0 + min(max(t, 0), 1) * (x1 - x0)
        g = "|" if per_row < 0.8 else ("/" if (x1 - x0) * (y1 - y0) < 0 else "\\")
        items.append(((r, math.floor(x)), g))
    emit(cells, items, dash)


def paint(g, cells, cls, box):
    for (r, c), ch in cells.items():
        if box[0] <= c <= box[2] and box[1] <= r <= box[3]:
            g.put(c, r, ch, cls)


# ---------------------------------------------------------------- the planes
VIEW = (1, 0, 46, 16)                     # clip box of the 3-D view (c0, r0, c1, r1)
C0 = 4                                    # column of the front-left corner
W = 28                                    # plane width in columns
D = 6                                     # plane depth in rows
SH = 2                                    # columns of shear per row of depth
YF = 14.95                                # row position of the front edge (bottom of row 14)
H_K, A_K, B_K = 5.0, 3.0, 3.5             # rows per unit of height offset / tilt a / tilt b


def pos(u, v, height=0.0):
    return C0 + 0.5 + W * u + SH * D * v, YF - D * v - height


def corner_height(state, u, v):
    ta, tb, h = state
    return H_K * h + A_K * ta * (2 * v - 1) + B_K * tb * (2 * u - 1)


def plane_cells(state, interior):
    """Outline and interior cells of the plane at `state` (None = the level target)."""
    def P(u, v):
        return pos(u, v, 0.0 if state is None else corner_height(state, u, v))

    out, inner = {}, {}
    thirds = (1 / 3, 2 / 3) if interior else ()
    for cells, vs, us in ((inner, thirds, thirds), (out, (0, 1), (0, 1))):
        for v in vs:
            col_line(cells, P(0, v), P(1, v))
        for u in us:
            row_line(cells, P(u, 0), P(u, 1))
    return out, inner


def target_cells():
    cells = {}
    for v in (0, 1):
        col_line(cells, pos(0, v), pos(1, v), dash=True)
    for u in (0, 1):
        row_line(cells, pos(u, 0), pos(u, 1), dash=True)
    for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = pos(u, v)
        r, _ = flat_glyph(y)
        cells[(r, math.floor(x))] = "+"
    return cells


def posts(g, state):
    """':' between each corner of the plane and the matching target corner."""
    for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, ya = pos(u, v, corner_height(state, u, v))
        _, yt = pos(u, v)
        lo, hi = min(ya, yt), max(ya, yt)
        for r in range(math.floor(lo), math.floor(hi) + 1):
            if lo + 0.35 < r + 0.5 < hi - 0.35 and 0 <= r < ROWS and g.ch[r, math.floor(x)] == " ":
                g.put(math.floor(x), r, ":", "t2")


# ---------------------------------------------------------------- plot
PX0, PX1 = 48, 67                         # x range of the plot area (columns, right edge exclusive)
PY0, PY1 = 2.1, 8.9                       # y range (rows): top, bottom


def plot_curve(curve):
    n = len(curve) - 1
    lo, hi = float(curve.min()), float(curve.max())
    xs = [PX0 + (PX1 - PX0) * i / n for i in range(n + 1)]
    ys = [PY1 - (PY1 - PY0) * (v - lo) / (hi - lo) for v in curve]

    def y_at(x):
        return float(np.interp(x, xs, ys))

    cells = {}
    for c in range(PX0, PX1):
        ya, yb, ym = y_at(c + 0.02), y_at(c + 0.98), y_at(c + 0.5)
        s = yb - ya
        if abs(s) <= 0.4:
            r, g = flat_glyph(ym)
            cells[(r, c)] = g
        elif abs(s) <= 1.5:
            cells[(math.floor(ym), c)] = "/" if s < 0 else "\\"
        else:
            lo_, hi_ = min(ya, yb), max(ya, yb)
            for r in range(math.floor(lo_), math.floor(hi_) + 1):
                if lo_ <= r + 0.5 <= hi_ or r == math.floor(ym):
                    cells[(r, c)] = "|"
    marks = [(min(math.floor(xs[i]), PX1 - 1), math.floor(ys[i])) for i in range(n + 1)]
    return cells, marks


def static_layer(curve):
    g = Grid(COLS, ROWS)
    for r in range(1, 9):
        g.put(PX0 - 1, r, "|", "t1")
    for c in range(PX0, PX1):
        g.put(c, 9, "-", "t1")
    g.put(PX0 - 1, 9, "+", "t1")
    g.text(PX0 - 1, 0, "return", "t2")
    g.text(PX0, 10, "0", "t1")
    g.rtext(PX1 - 1, 10, "600", "t1")
    cells, marks = plot_curve(curve)
    paint(g, cells, "t3", (0, 0, COLS - 1, ROWS - 1))
    return g, marks


# ---------------------------------------------------------------- build
def build():
    snaps, curve = train()
    picks = [0, MID, len(curve) - 1]
    print("curve", np.round(curve[[0, 5, 10, 20, 30, 40, 50, 60]], 2), "picks", picks)

    base, marks = static_layer(curve)
    paint(base, target_cells(), "k", VIEW)

    frames = []
    for idx in picks:
        ret, states = rollout(snaps[idx], 11, trace=True)
        errs = [misalignment(s) for s in states]
        hit = next((i for i in range(len(errs) - 3) if max(errs[i:i + 4]) < TOL), None)
        print(f"policy@{idx}: return {ret:.1f}, final err {errs[-1]:.3f}, max err after 12: "
              f"{max(errs[12:]):.3f}, aligned at {hit}")
        episode = idx * 10
        mc, mr = marks[idx]
        seq = [0] + list(range(len(states))) + [len(states) - 1]          # hold, play, hold
        hit_f = None if hit is None else 1 + hit
        for f, si in enumerate(seq):
            s = states[si]
            g = base.copy()
            lit = hit_f is not None and hit_f <= f < hit_f + 7
            posts(g, s)
            out, inner = plane_cells(s, interior=True)
            paint(g, inner, "a" if lit else "t3", VIEW)
            paint(g, out, "a" if lit else "t4", VIEW)
            g.text(C0 + W + 3, 14, "target", "t2")
            if hit_f is not None and f >= hit_f:
                g.ctext(C0 + (W + SH * D) // 2, 16, "aligned", "a")
            g.put(mc, mr, "o", "t5")
            g.text(PX0 - 1, 12, "err", "t2")
            g.text(PX0 + 3, 12, f"{errs[si]:.2f}", "a" if lit else "t3")
            g.text(PX0 - 1, 14, "episode", "t2")
            g.text(PX0 + 7, 14, f"{episode:04d}", "t3")
            frames.append(g)

    m = Movie(COLS, ROWS, title="A wireframe plane learning to align",
              desc="A wireframe plane learns to align with a fixed dashed target plane. Three replayed rollouts "
                   "from the same misaligned pose, early, middle and late in training: the early policy wobbles "
                   "and overshoots, the middle one reaches the target with some overshoot and the late one "
                   "settles smoothly, flashing red with an aligned tag. A learning curve with a marker and a "
                   "live error readout sit on the right.")
    m.add(frames, FPS)
    return m, len(frames)


if __name__ == "__main__":
    m, n = build()
    per = n // 3
    print("frames", n, "seconds", n / FPS)
    if "--preview" in sys.argv:
        ts = [0.1, 1.3, 2.6, per / FPS + 0.1, per / FPS + 1.3, per / FPS + 2.6,
              2 * per / FPS + 0.1, 2 * per / FPS + 1.2, 2 * per / FPS + 2.4, (n - 1) / FPS]
        m.preview("alignment.png", times=ts)
        m.preview("alignment-light.png", times=(1.3, 2 * per / FPS + 2.4), theme="light")
    m.write("alignment.svg")
