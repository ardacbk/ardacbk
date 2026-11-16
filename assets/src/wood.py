"""Wood: one board, inspected three times.

A planed board with fibre lines bending around a natural knot (concentric
rings, not a defect) and a thin jagged crack (the defect) is rasterised into
a real float image (4x8 samples per character cell, on the fine grid). The
board is printed from that image: a light, grained surface in which the knot's
rings and the crack show as dark breaks. Three small, genuinely different
detectors stand in for the three real models:

  GLASS        local deviation from a blurred reference
  EfficientAD  patch difference: each patch against its neighbours along the grain
  INP-Former   inconsistency of the local gradient orientation (structure tensor)

Each map is scaled by what the same detector says about a healthy board (the
same board without the crack) times a fixed sensitivity. A scanner sweeps the
board once per model; behind it the map is painted over the grain, held with a
score readout, then thresholded away again.
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter

from engine import CARD_FINE, FINE, SX, SY, Grid, Movie, Printer, cell_max, cell_mean

COLS, ROWS = CARD_FINE
BC0, BR0, BCOLS, BROWS = 1, 2, 93, 20          # board interior (cells)
H, W = BROWS * SY, BCOLS * SX                   # 160 x 372 samples
FX, FY = W / 264, H / 112                       # the geometry was laid out on a 264 x 112 board
FPS = 10
PHASE = 36                                      # frames per model
N = 3 * PHASE
SWEEP, HOLD, FADE = 14, 11, 6                   # then clean frames before the next phase
LEAD = 2                                        # clean frames before the scanner enters

KNOT = (70.0 * FX, 44.0 * FY, 32.0 * FX, 19.0 * FY)   # cx, cy, rx, ry in samples
KNOT_RINGS = (0.40, 0.72, 1.0)
CRACK_X0, CRACK_X1 = int(160 * FX), int(220 * FX)     # samples
CRACK_Y = 76 * FY
MODELS = ["GLASS", "EfficientAD", "INP-Former"]
BS = "\\"


# ---------------------------------------------------------------- the board geometry
def crack_path(rng):
    xs = np.arange(CRACK_X0, CRACK_X1)
    y = np.empty(len(xs))
    cur = float(CRACK_Y)
    for i in range(len(xs)):
        cur += rng.normal(0, 0.3)
        if rng.random() < 0.06:
            cur += rng.choice([-1, 1]) * rng.uniform(5, 10)     # a jag
        cur += 0.05 * (CRACK_Y - cur)
        y[i] = cur
    return xs, y


def fibres(seed=11):
    """Fibre lines: y(x) per sample column, a live mask per cell column, a colour class."""
    rng = np.random.default_rng(seed)
    kx, ky, rx, ry = KNOT
    xs = np.arange(W) + 0.5
    base = gaussian_filter1d(rng.standard_normal(W), 35, mode="reflect")
    base = 2.6 * base / base.std()
    out = []
    for i in range(BROWS):
        y0 = 4.0 + SY * i + rng.uniform(-1, 1)
        band = min(abs(y0 - 24 * FY), abs(y0 - 92 * FY)) < 10
        skip = rng.random() < 0.28 and not band and abs(y0 - ky) > ry * 1.2
        w = gaussian_filter1d(rng.standard_normal(W), 28, mode="reflect")
        w = np.clip(0.8 * base + w / w.std(), -3.0, 3.0) * 0.8
        d0 = y0 - ky
        g = np.maximum(0, 1 - ((xs - kx) / rx) ** 2)
        side = 1 if d0 >= -0.5 else -1
        y = ky + side * np.sqrt(d0 ** 2 + (1.2 * ry) ** 2 * g) + w * (1 - 0.7 * g)
        live = np.ones(BCOLS, bool)
        if rng.random() < 0.5:
            live[:int(rng.integers(0, 8))] = False
        if rng.random() < 0.5:
            live[BCOLS - int(rng.integers(1, 10)):] = False
        c = 3
        while c < BCOLS:
            if rng.random() < 0.075:
                n = int(rng.integers(2, 7))
                live[c:c + n] = False
                c += n + 4
            c += 1
        if skip:
            live[:] = False
        out.append(dict(y=y, live=live, cls="t2" if band else "t1", amp=0.16 if band else 0.10))
    return out, rng


def board(with_crack=True):
    """Float image of the board (0 dark .. 1 light), shape (H, W)."""
    fib, _ = fibres()
    yy = np.arange(H, dtype=np.float64)[:, None] + 0.5
    kx, ky, rx, ry = KNOT
    rng = np.random.default_rng(21)
    img = 0.66 + 0.10 * gaussian_filter(rng.standard_normal((H, W)), (1, 3))
    for f in fib:
        live = np.repeat(f["live"], SX)
        img -= f["amp"] * np.exp(-((yy - f["y"][None, :]) / 1.2) ** 2) * live[None, :]
    xx = np.arange(W)[None, :] + 0.5
    rho = np.sqrt(((xx - kx) / rx) ** 2 + ((yy - ky) / ry) ** 2)
    img -= 0.16 * np.clip(1.15 - rho, 0, 1) ** 0.5                  # the knot is darker wood
    for r0 in KNOT_RINGS:                                            # thick enough to survive the print
        img -= 0.22 * np.exp(-((rho - r0) * ry / 3.0) ** 2)
    img -= 0.3 * np.exp(-(rho / 0.18) ** 2)
    path = None
    if with_crack:
        xs, cy = crack_path(np.random.default_rng(5))
        path = (xs, cy)
        for x, c in zip(xs, cy):
            taper = min(1, (x - CRACK_X0) / 6, (CRACK_X1 - 1 - x) / 6) ** 0.6
            img[:, x] -= 0.75 * taper * np.exp(-((yy[:, 0] - c) / 3.2) ** 2)   # about a cell tall
    return np.clip(img, 0, 1), path


# ---------------------------------------------------------------- three detectors
def pad(img, k=16):
    return np.pad(img, k, mode="reflect"), k


def glass(img):
    """local deviation from a blurred reference"""
    p, k = pad(img)
    dev = np.abs(p - gaussian_filter(p, 4.0))
    return gaussian_filter(dev, 1.5)[k:-k, k:-k]


def efficientad(img):
    """patch difference: every patch against its neighbours along the grain"""
    p, k = pad(img)
    d = p - 0.5 * (np.roll(p, 9, 1) + np.roll(p, -9, 1))
    return np.sqrt(uniform_filter(d * d, (7, 10)))[k:-k, k:-k]


def inp_former(img):
    """inconsistency of the gradient orientation: 1 - coherence, weighted by gradient energy"""
    p, k = pad(img)
    gy, gx = np.gradient(gaussian_filter(p, 1.0))
    jxx, jyy, jxy = (gaussian_filter(a, 3.5) for a in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-9)
    return ((1 - coh) * np.sqrt(tr))[k:-k, k:-k]


DETECTORS = [glass, efficientad, inp_former]
K_SCALE = (3.3, 2.9, 4.0)


def maps():
    img, path = board(True)
    healthy, _ = board(False)
    out, scores = [], []
    for det, k in zip(DETECTORS, K_SCALE):
        ref = det(healthy).max()
        m = det(img) / (k * ref)
        out.append(m)
        scores.append(min(m.max(), 0.99))
    return img, path, out, scores


# ---------------------------------------------------------------- printing the board
PRINT = Printer(edge=0.28, floor=0.05)


def board_grid(img):
    """The board image as glyphs and tones: wood is light, so it prints dense; dark breaks print sparse."""
    v = np.clip(0.42 + 1.1 * (img - 0.66), 0, 1)        # wood mid-dense; fibres, rings and the crack darker
    ch, lum = PRINT(v)
    cl = np.full(ch.shape, "t3", dtype="<U2")
    cl[lum < 0.3] = "t2"
    return ch, cl


# ---------------------------------------------------------------- composition
def build():
    img, path, mp, scores = maps()
    for n, m, s in zip(MODELS, mp, scores):
        cm = cell_max(m)
        kr, kc = int(KNOT[1] // SY), int(KNOT[0] // SX)
        kn = cm[max(kr - 3, 0):kr + 4, kc - 9:kc + 10]
        print(f"{n:12s} max {m.max():.2f}  knot-area cell max {kn.max():.2f}  >0.46: {(cm >= 0.46).sum()} cells")

    kr, kc = int(KNOT[1] // SY), int(KNOT[0] // SX)
    base = Grid(COLS, ROWS)
    base.text(BC0, 1, "_" * BCOLS, "t2")
    base.text(BC0, BR0 + BROWS, "-" * BCOLS, "t2")
    bch, bcl = board_grid(img)
    bcl[bch == " "] = ""
    base.paste(bch, bcl, BC0, BR0)
    print("empty board cells: %.0f%%" % (100 * (base.ch[BR0:BR0 + BROWS, BC0:BC0 + BCOLS] == " ").mean()))

    cmaps = [cell_max(m) for m in mp]
    A_THR, B_THR = 0.62, 0.46

    def overlay(g, m, rel):
        """paint map m (cells) over the grain; rel in 0..1 is how much has faded"""
        lo = B_THR + (1.2 - B_THR) * rel
        hi = A_THR + (1.4 - A_THR) * rel
        for r in range(BROWS):
            for c in range(BCOLS):
                v = m[r, c]
                if v < lo:
                    continue
                if v >= hi:
                    ch = "@" if v >= 0.95 else ("%" if v >= 0.78 else "#")
                    g.put(BC0 + c, BR0 + r, ch, "a")
                else:
                    ch = "*" if v >= 0.5 else ("+" if v >= 0.44 else ":")
                    g.put(BC0 + c, BR0 + r, ch, "b")

    frames = []
    for f in range(N):
        k, p = divmod(f, PHASE)
        g = Grid(COLS, ROWS)
        m = cmaps[k]
        done_at = LEAD + SWEEP
        active = None
        score_shown = tags = False
        if LEAD <= p < done_at:
            active = k
            front = int((p - LEAD + 1) / SWEEP * (BCOLS + 1))
            sc = min(front, BCOLS)
            overlay(g, np.where(np.arange(BCOLS)[None, :] < sc - 1, m, 0), 0)
            if 0 <= front - 1 < BCOLS:
                for r in range(BROWS):
                    g.put(BC0 + front - 1, BR0 + r, "|", "t5")
        elif done_at <= p < done_at + HOLD:
            active = k
            overlay(g, m, 0)
            score_shown = tags = True
        elif done_at + HOLD <= p < done_at + HOLD + FADE:
            active = k
            overlay(g, m, (p - done_at - HOLD + 1) / FADE)
            score_shown = True
        x = 2
        for i, name in enumerate(MODELS):
            g.text(x, 0, name, "t5" if active == i else "t2")
            x += len(name) + 3
        if score_shown:
            g.rtext(COLS - 2, 0, f"score {scores[k]:.2f}", "t5")
        if tags:
            g.ctext(BC0 + kc, ROWS - 2, "knot", "t2")
            g.ctext(BC0 + (CRACK_X0 + CRACK_X1) // 2 // SX, ROWS - 2, "crack", "t2")
        frames.append(g)

    m = Movie(COLS, ROWS, cell=FINE, title="Anomaly maps on a wooden board",
              desc="A planed wooden board with a natural knot and a crack is scanned three times, once each by "
                   "GLASS, EfficientAD and INP-Former. Each sweep paints that model's anomaly map in red over "
                   "the grain: it lights up on the crack and stays quiet on the knot, then fades out with a score.")
    m.add(base)
    m.add(frames, FPS, opaque=True)
    return m


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("wood.png", times=(0.0, 1.0, 2.6, 3.5, 5.4, 8.4, 10.5))
        m.preview("wood-light.png", times=(0.0, 3.5), theme="light")
    m.write("wood.svg")
