"""Leaves: a leaf getting sick before it looks sick.

A procedurally drawn leaf (outline, midrib, curved side veins) is rendered as a
float image at 4 x 8 samples per cell and turned into line glyphs.  On that same
sample grid a lesion field L is simulated: three infection sites, logistic growth
plus diffusion (Fisher-KPP), the growth rate depending on air humidity.

Two sensor traces come out of the same simulation: humidity (a scripted weather
event that rises for a while) and leaf temperature (drifts up with the infected
mass).  A small fusion model reads one image feature, the lesion area fraction
measured from L, plus the two sensor features and returns p(disease) through a
logistic function.  The sensors push p over 0.5 while the leaf still looks
almost clean; the spots only become obvious afterwards.  Then a new sample is
loaded: a wipe redraws everything from the top, back to the first frame.
"""
import math
import sys

import numpy as np

from engine import CARD, SX, SY, Grid, Movie, cell_mean

COLS, ROWS = CARD
NX, NY = COLS * SX, ROWS * SY
FPS = 8                     # traces and wipe; leaf and meter run at half that
N_SIM = 88                  # frames of the episode (11 s)
N_WIPE = 10                 # frames of the reload sweep (1.25 s)
STEPS = 6                   # diffusion steps per frame
WIN = 23                    # trace width in cells
PC = 44                     # first column of the sensor panel

# ---------------------------------------------------------------- leaf geometry (in cells)
C0, LL, RM = 6, 36, 9       # base column, length in columns, midrib row
HMAX = 5.6                  # half height in rows at the widest point
VEINS = [4, 10, 16, 22, 28]


def half_h(x):
    """half height (rows) of the blade at x columns from the base: ovate, pointed tip"""
    t = np.clip(np.asarray(x, float) / LL, 0, 1)
    peak = 0.36 ** 0.55 * 0.64
    return HMAX * t ** 0.55 * (1 - t) / peak


def leaf_mask():
    """blade on the sample grid"""
    yy, xx = np.mgrid[:NY, :NX] + 0.5
    xc, yc = xx / SX - C0, yy / SY
    return (xc > 0) & (xc < LL) & (np.abs(yc - (RM + 0.5)) < half_h(xc) - 0.5)


def inside_cells():
    rr, cc = np.mgrid[:ROWS, :COLS]
    xc = cc + 0.5 - C0
    return (xc > 0) & (xc < LL) & (np.abs(rr + 0.5 - (RM + 0.5)) < half_h(xc) - 0.9)


def edge_glyph(p, upper):
    """a flat bit of edge at row-coordinate p -> (row, glyph); one glyph per column"""
    r, fr = int(math.floor(p)), p - math.floor(p)
    if fr < 0.25:
        return r - 1, "_"
    return (r, "-") if fr < 0.7 else (r, "_")


def leaf_glyphs():
    """static characters of the clean leaf: (chars, classes, blade mask)"""
    ch = np.full((ROWS, COLS), " ", dtype="<U1")
    cl = np.full((ROWS, COLS), "", dtype="<U2")

    def put(c, r, g, k, force=False):
        if 0 <= r < ROWS and 0 <= c < COLS and (force or ch[r, c] == " "):
            ch[r, c], cl[r, c] = g, k

    mid = RM + 0.5
    used = {True: set(), False: set()}
    # outline: the upper edge rises then falls; the lower one mirrors it
    for c in range(C0, C0 + LL):
        h0, h1 = float(half_h(c - C0)), float(half_h(c + 1 - C0))
        hm = float(half_h(c + 0.5 - C0))
        rising = h1 > h0
        for upper in (True, False):
            sgn = -1 if upper else 1
            p0, p1, pm = mid + sgn * h0, mid + sgn * h1, mid + sgn * hm
            if abs(hm) < 0.45:                                   # edges have met the midrib
                continue
            if abs(h1 - h0) <= 0.62:
                r, g = edge_glyph(pm, upper)
                if r != RM:
                    put(c, r, g, "t4")
            else:
                lo, hi = sorted((p0, p1))
                rows = range(int(math.floor(lo)), int(math.floor(hi)) + 1)
                for r in rows:
                    if r == RM or r in used[upper]:
                        continue
                    used[upper].add(r)
                    if c == C0:
                        g = "("
                    elif upper:
                        g = "/" if rising else "\\"
                    else:
                        g = "\\" if rising else "/"
                    put(c, r, g, "t4")
    put(C0 + LL - 1, RM, ">", "t4", True)
    for c in range(C0 - 4, C0):                                  # stem
        put(c, RM, "-", "t4", True)
    # midrib and side veins (leaning toward the tip, stopping a cell short of the edge)
    for c in range(C0, C0 + LL - 1):
        put(c, RM, "-", "t3", True)
    for xk in VEINS:
        c0 = C0 + xk
        for j in range(1, 8):
            c = c0 + j
            if c >= C0 + LL - 2:
                break
            h = float(half_h(c + 0.5 - C0))
            if RM - j < math.floor(mid - h) + 2:
                break
            put(c, RM - j, "/", "t2")
        for j in range(1, 8):
            c = c0 + j
            if c >= C0 + LL - 2:
                break
            h = float(half_h(c + 0.5 - C0))
            if RM + j > math.floor(mid + h) - 2:
                break
            put(c, RM + j, "\\", "t2")
    # sparse tissue texture
    inside = inside_cells()
    rng = np.random.default_rng(5)
    tex = (rng.random((ROWS, COLS)) < 0.125) & inside & (ch == " ")
    ch[tex], cl[tex] = ".", "t1"
    return ch, cl, inside


# ---------------------------------------------------------------- simulation
def smooth_noise(n, seed, k=9, amp=1.0):
    r = np.random.default_rng(seed).normal(size=n + 2 * k)
    s = np.convolve(r, np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), "same")[k:-k]
    return amp * s / s.std()


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lap(a):
    p = np.pad(a, 1, mode="edge")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * a


def logistic(x):
    return 1 / (1 + math.exp(-x))


PRE = 100   # frames of baseline history shown before frame 0 (so the traces start full)


def simulate():
    mask = leaf_mask()
    # keep lesions away from the very edge
    inner = mask.copy()
    for _ in range(5):
        inner = inner & np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
    yy, xx = np.mgrid[:NY, :NX] + 0.5
    L = np.zeros((NY, NX), np.float32)
    for (c, r, amp) in ((C0 + 16, RM - 3, 0.012), (C0 + 19, RM + 2, 0.004), (C0 + 10, RM + 3, 0.0025)):
        x, y = (c + 0.5) * SX, (r + 0.5) * SY
        L += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
    L *= inner

    total = PRE + N_SIM
    hn = smooth_noise(total, 1, amp=0.012)
    tn = smooth_noise(total, 2, amp=0.07)
    f = np.arange(total) - PRE
    hum = 0.46 + 0.43 * smoothstep((f - 4) / 30) + hn                     # a damp spell
    hum = np.where(f < 0, 0.46 + hn, hum)
    D = 0.12
    area, mass, Ls = [], [], []
    temp = np.zeros(total)
    p = np.zeros(total)
    cur_dT = 0.0
    for i in range(total):
        h = hum[i]
        r_g = 0.003 + 0.022 * float(np.clip((h - 0.52) / 0.33, 0, 1)) ** 1.5      # humid -> fast spread
        if f[i] >= 0:
            for _ in range(STEPS):
                L = L + D * lap(L) + r_g * L * (1 - L)
                L = np.clip(L * inner, 0, 1)
        m = float(L.sum() / inner.sum())
        a = float((L > 0.3).sum() / inner.sum())
        target = 3.2 * (1 - math.exp(-m / 0.010))                                    # leaf temp follows infected mass
        cur_dT += 0.18 * (target - cur_dT)
        temp[i] = cur_dT + tn[i]
        hf = (h - 0.46) / 0.43
        tf = np.clip(cur_dT / 3.0, 0, 1)
        z = -4.2 + 3.1 * hf + 2.6 * tf + 24.0 * a                                    # the fusion model
        p[i] = logistic(z)
        area.append(a)
        mass.append(m)
        Ls.append(L.copy() if i >= PRE else None)
    return dict(hum=hum, temp=temp, p=p, area=np.array(area[PRE:]), L=Ls[PRE:], mask=mask)


# ---------------------------------------------------------------- drawing
def glyphs_from_mask(m):
    r"""boolean sample mask (rows*8, cols*4) -> line glyphs by local slope"""
    rows, cols = m.shape[0] // SY, m.shape[1] // SX
    ch = np.full((rows, cols), " ", dtype="<U1")
    for r in range(rows):
        for c in range(cols):
            blk = m[r * SY:(r + 1) * SY, c * SX:(c + 1) * SX]
            if blk.sum() < 2:
                continue
            ys, xs = np.nonzero(blk)
            ext = ys.max() - ys.min() + 1
            if ext >= 7 and np.ptp(xs) <= 1:
                g = "|"
            elif ext >= 4:
                lft = ys[xs <= 1].mean() if (xs <= 1).any() else ys.mean()
                rgt = ys[xs >= 2].mean() if (xs >= 2).any() else ys.mean()
                g = "/" if rgt < lft else "\\"
            else:
                fy = ys.mean() / SY
                if fy < 0.34 and r > 0:
                    if ch[r - 1, c] == " ":
                        ch[r - 1, c] = "_"
                    continue
                g = "_" if fy >= 0.7 else "-"
            if ch[r, c] == " ":
                ch[r, c] = g
    return ch


def trace_cells(series, lo, hi, rows=4):
    """last WIN*4 samples of a series -> (chars, strength) on a rows x WIN cell patch, drawn as line art"""
    h = rows * SY
    n = WIN * SX
    y = (1 - np.clip((series - lo) / (hi - lo), 0, 1)) * (h - 3) + 1.5
    m = np.zeros((h, n), bool)
    for j in range(n):
        a = y[j]
        b = y[min(j + 1, n - 1)]
        y0, y1 = sorted((a, b))
        m[int(y0):int(y1) + 1, j] = True
    return glyphs_from_mask(m)


def lesion_overlay(Lc, ch0, cl0, inside):
    """characters for the stress field; only blade cells, never the outline"""
    ch, cl = ch0.copy(), cl0.copy()
    tex = ch0 == "."
    barely = (Lc >= 0.03) & (Lc < 0.12) & tex
    ch[barely], cl[barely] = ch0[barely], "b"
    for t, g, c in [(0.12, ".", "b"), (0.2, ":", "b"), (0.3, "o", "b"), (0.45, "*", "b"), (0.6, "*", "a"), (0.75, "@", "a")]:
        sel = (Lc >= t) & inside
        ch[sel], cl[sel] = g, c
    return ch, cl


def put_traces(g, sim, i):
    j = PRE + i
    for (r0, series, lo, hi) in ((2, sim["hum"], 0.40, 0.96), (8, sim["temp"], -0.5, 3.6)):
        ch = trace_cells(series[j - WIN * SX + 1:j + 1], lo, hi)
        cl = np.where(ch != " ", "t3", "")
        cl[:, -1] = np.where(ch[:, -1] != " ", "t4", "")
        g.paste(ch, cl, PC, r0)


def put_meter(g, sim, i):
    pv = sim["p"][PRE + i]
    hot = pv >= 0.5
    cls = "a" if hot else "t3"
    g.text(PC, 1, "humidity", "t2")
    g.text(PC, 7, "leaf temp", "t2")
    g.text(PC, 13, "p(disease)", "t2")
    k = int(round(pv * 12))
    g.text(PC, 14, "[" + "#" * k + "." * (12 - k) + "]", cls)
    g.text(PC + 14, 14, f"{pv:.2f}", cls)
    if hot:
        g.text(PC, 15, "early warning", "a")


def build():
    sim = simulate()
    ch0, cl0, inside = leaf_glyphs()
    area, p = sim["area"], sim["p"][PRE:]
    cross = int(np.argmax(p >= 0.5))
    first_o = next(i for i in range(N_SIM) if (cell_mean(sim["L"][i]) >= 0.3).any())
    print(f"p crosses 0.5 at frame {cross} (A={area[cross]:.4f}); first 'o' spot at frame {first_o}; "
          f"A end {area[-1]:.3f}; p0={p[0]:.2f} p_end={p[-1]:.2f}")

    def episode(i):
        g = Grid(COLS, ROWS)
        ch, cl = lesion_overlay(cell_mean(sim["L"][i]), ch0, cl0, inside)
        g.paste(ch, cl)
        put_meter(g, sim, i)
        return g

    def tr(i):
        g = Grid(COLS, ROWS)
        put_traces(g, sim, i)
        return g

    A = [episode(i) for i in range(0, N_SIM, 2)]
    B = [tr(i) for i in range(N_SIM)]
    new = Grid(COLS, ROWS)                      # the next sample: the very first frame
    new.paste(A[0].ch, A[0].cl)
    new.paste(B[0].ch, B[0].cl)
    A += [A[-1]] * (N_WIPE // 2)                # the old sample waits under the sweep
    B += [B[-1]] * N_WIPE
    W = [Grid(COLS, ROWS) for _ in range(N_SIM)]
    for k in range(N_WIPE):
        g = Grid(COLS, ROWS)
        cw = round((k + 1) * COLS / N_WIPE) if k < N_WIPE - 1 else COLS
        g.ch[:, :cw], g.cl[:, :cw] = new.ch[:, :cw], new.cl[:, :cw]
        g.cover(0, 0, cw - 1, ROWS - 1)
        if cw < COLS:
            for r in range(1, ROWS - 1):
                g.put(cw, r, "|", "t3")
        W.append(g)

    m = Movie(COLS, ROWS, pad=(14, 12), title="A leaf getting sick before it looks sick",
              desc="An ASCII leaf drawn as a clean line drawing, with two scrolling sensor traces "
                   "(humidity, leaf temperature) and a p(disease) meter. As a simulated lesion field grows from "
                   "three infection sites, the meter fed by image and sensor evidence turns red with an early "
                   "warning while the leaf still looks almost clean; only later do the spots show. Then a new "
                   "sample is swept in from the left.")
    m.add(A, FPS // 2)
    m.add(B, FPS)
    m.add(W, FPS, opaque=True)
    return m


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("leaves.png", times=(0, 3.0, 5.0, 6.0, 7.0, 8.5, 10.9, 11.5))
        m.preview("leaves-light.png", times=(6.0, 10.0), theme="light")
    m.write("leaves.svg")
