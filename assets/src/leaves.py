"""Leaves: a leaf getting sick before it looks sick.

A procedurally drawn leaf (outline, midrib, curved side veins) is rendered as a
shaded float image at 4 x 8 samples per cell on the fine grid and printed with
the shared Printer: a lit blade whose veins and rim show as brighter lines.
On that same sample grid a lesion field L is simulated: three infection sites, logistic growth
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

from engine import CARD_FINE, FINE, SX, SY, Grid, Movie, Printer, cell_mean

COLS, ROWS = CARD_FINE
NX, NY = COLS * SX, ROWS * SY
FPS = 8                     # traces and wipe; leaf and meter run at half that
N_SIM = 88                  # frames of the episode (11 s)
N_WIPE = 10                 # frames of the reload sweep (1.25 s)
STEPS = 8                   # diffusion steps per frame
WIN = 30                    # trace width in cells
PC = 62                     # first column of the sensor panel

# ---------------------------------------------------------------- leaf geometry (in cells)
C0, LL, RM = 7, 50, 12      # base column, length in columns, midrib row
HMAX = 8.0                  # half height in rows at the widest point
VEINS = [5, 13, 21, 29, 37]


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


PRINT = Printer(edge=0.3, floor=0.05)


def leaf_image():
    """The clean leaf as a shaded image (NY, NX): a blade lit across its width,
    brighter veins and rim, and a short stem."""
    yy, xx = np.mgrid[:NY, :NX] + 0.5
    xc, yc = xx / SX - C0, yy / SY - (RM + 0.5)            # in cells, from the base and the midrib
    hh = half_h(xc)
    inside = (xc > 0) & (xc < LL) & (np.abs(yc) < hh)
    u = np.clip(np.abs(yc) / np.maximum(hh, 1e-3), 0, 1)   # 0 at the midrib, 1 at the edge
    img = np.where(inside, 0.14 + 0.14 * (1 - u) - 0.05 * (yc > 0), 0.0)
    rng = np.random.default_rng(5)
    img += inside * 0.03 * rng.standard_normal((NY, NX))
    # rim: a thin band just inside the edge
    line = inside * np.exp(-((hh - np.abs(yc)) * SY / 1.3) ** 2)
    img = np.maximum(img, 0.7 * line)
    return np.clip(img, 0, 1), inside


def vein_glyphs():
    """Midrib and side veins as line glyphs placed by slope (they are thin, so
    drawing them directly reads better than printing them from the image)."""
    out = {}
    for c in range(C0 - 4, C0 + LL - 1):
        out[(RM, c)] = ("-" if c >= C0 else "=", "t4")
    for xk in VEINS:
        for sgn in (-1, 1):
            n = 13
            reach = min(0.7 * float(half_h(xk + n)), n * 0.45 * SX / SY)
            ys = [RM + 0.5 + sgn * reach * (1 - (1 - k / n) ** 1.6) for k in range(n + 1)]
            for k in range(1, n):
                c = C0 + xk + k
                r = int(np.floor(ys[k]))
                if r == RM or abs(ys[k] - (RM + 0.5)) > float(half_h(xk + k)) - 1.2:
                    continue
                nr = int(np.floor(ys[k + 1]))
                if nr != r:
                    g = "/" if sgn < 0 else "\\"
                else:
                    g = "_" if ys[k] - r >= 0.62 else "-"
                out.setdefault((r, c), (g, "t3"))
    return out


VEINS_AT = None


def put_veins(ch, cl, keep):
    global VEINS_AT
    if VEINS_AT is None:
        VEINS_AT = vein_glyphs()
    for (r, c), (g, k) in VEINS_AT.items():
        if 0 <= r < ROWS and 0 <= c < COLS and keep[r, c]:
            ch[r, c], cl[r, c] = g, k


def leaf_glyphs():
    """static characters of the clean leaf: (chars, classes, blade mask per cell, image)"""
    img, inside = leaf_image()
    ch, lum = PRINT(img)
    cl = np.full(ch.shape, "t2", dtype="<U2")
    cl[lum > 0.2] = "t3"
    cl[lum > 0.35] = "t4"
    put_veins(ch, cl, np.ones(ch.shape, bool))
    cl[ch == " "] = ""
    return ch, cl, cell_mean(inside.astype(float)) > 0.5, img


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


PRE = 130   # frames of baseline history shown before frame 0 (so the traces start full)


def simulate():
    mask = leaf_mask()
    # keep lesions away from the very edge
    inner = mask.copy()
    for _ in range(5):
        inner = inner & np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
    yy, xx = np.mgrid[:NY, :NX] + 0.5
    L = np.zeros((NY, NX), np.float32)
    for (c, r, amp) in ((C0 + 23, RM - 4, 0.012), (C0 + 27, RM + 3, 0.004), (C0 + 14, RM + 4, 0.0025)):
        x, y = (c + 0.5) * SX, (r + 0.5) * SY
        L += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.8 ** 2))
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


def trace_cells(series, lo, hi, rows=6):
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


def lesion_overlay(L, img0, inside):
    """The leaf with its lesions: a yellowing halo (dim red tone, the leaf's own
    glyphs) around a necrotic core printed dense in the accent."""
    core = np.clip((L - 0.35) / 0.4, 0, 1)
    img = img0 * (1 - core) + 0.95 * core                  # the spot itself prints dense
    ch, lum = PRINT(img)
    Lc = cell_mean(L)
    cl = np.full(ch.shape, "t2", dtype="<U2")
    cl[lum > 0.2] = "t3"
    cl[lum > 0.35] = "t4"
    put_veins(ch, cl, Lc < 0.45)
    cl[(Lc >= 0.06) & inside] = "b"
    cl[(Lc >= 0.45) & inside] = "a"
    cl[ch == " "] = ""
    return ch, cl


def put_traces(g, sim, i):
    j = PRE + i
    for (r0, series, lo, hi) in ((2, sim["hum"], 0.40, 0.96), (11, sim["temp"], -0.5, 3.6)):
        ch = trace_cells(series[j - WIN * SX + 1:j + 1], lo, hi)
        cl = np.where(ch != " ", "t3", "")
        cl[:, -1] = np.where(ch[:, -1] != " ", "t4", "")
        g.paste(ch, cl, PC, r0)


def put_meter(g, sim, i):
    pv = sim["p"][PRE + i]
    hot = pv >= 0.5
    cls = "a" if hot else "t3"
    g.text(PC, 1, "humidity", "t2")
    g.text(PC, 10, "leaf temp", "t2")
    g.text(PC, 19, "p(disease)", "t2")
    k = int(round(pv * 20))
    g.text(PC, 20, "[" + "#" * k + "." * (20 - k) + "]", cls)
    g.text(PC + 23, 20, f"{pv:.2f}", cls)
    if hot:
        g.text(PC, 22, "early warning", "a")


def build():
    sim = simulate()
    ch0, cl0, inside, img0 = leaf_glyphs()
    area, p = sim["area"], sim["p"][PRE:]
    cross = int(np.argmax(p >= 0.5))
    first_o = next(i for i in range(N_SIM) if (cell_mean(sim["L"][i]) >= 0.3).any())
    print(f"p crosses 0.5 at frame {cross} (A={area[cross]:.4f}); first 'o' spot at frame {first_o}; "
          f"A end {area[-1]:.3f}; p0={p[0]:.2f} p_end={p[-1]:.2f}")

    def episode(i):
        g = Grid(COLS, ROWS)
        ch, cl = lesion_overlay(sim["L"][i], img0, inside)
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

    m = Movie(COLS, ROWS, pad=(14, 12), cell=FINE, title="A leaf getting sick before it looks sick",
              desc="A shaded ASCII leaf with bright veins, with two scrolling sensor traces "
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
