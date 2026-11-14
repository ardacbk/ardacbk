"""Hero: the name standing in a ripple tank.

A 2-D wave equation is solved on a grid four times finer than the character
grid across and eight times down. A steady source hums at 0.67 Hz in the gap
between the two words; the letters are hard walls, so the waves reflect off
them and diffract through the gaps. Once the field has settled it repeats
exactly every period, so one period of frames loops forever.

Once per ten seconds a second source fires a short burst at a different
frequency somewhere else in the tank. It is simulated separately (the
equation is linear, so the two fields simply add) and drawn on top in the
accent colour: the thing that is off.

Crests are drawn as line art. The phase of the field comes from its analytic
signal, a crest is where that phase crosses zero, and each character cell
the crest passes through gets _ - / \\ or | depending on the local slope.
"""
import math
import sys

import numpy as np

from engine import SX, SY, Grid, Movie, cell_mean

COLS, ROWS = 140, 28
NX, NY = COLS * SX, ROWS * SY
PADDING = 56                      # absorbing sponge around the visible tank
FPS = 12
PERIOD_FRAMES = 18                # one hum period = 1.5 s -> 0.67 Hz
STEPS_PER_FRAME = 3               # wave speed 0.5 sample/step -> wavelength 27 samples
C2 = 0.25                         # (c*dt/dx)^2
LOOP_FRAMES = 120                 # burst + readout loop: 10 s
HUD_FPS = 4
BURST_AT = 24                     # frame the burst starts
BURST_WAVELENGTH = 19             # samples -> 0.67 Hz * 27 / 19 = 0.95 Hz

# ---------------------------------------------------------------- the name
PIXELS = {  # 5 x 9 pixel font; one pixel = 2 columns x 1 row, which is close to square
    "A": [".###.", "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#", "#...#"],
    "R": ["####.", "#...#", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#", "#...#"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "Ç": [".###.", "#...#", "#....", "#....", "#....", "#....", "#....", "#...#", ".###.", "..#..", ".##.."],
    "B": ["####.", "#...#", "#...#", "#...#", "####.", "#...#", "#...#", "#...#", "####."],
    "U": ["#...#"] * 8 + [".###."],
    "K": ["#...#", "#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#", "#...#"],
}
NAME, NAME_ROW = "ARDA ÇABUK", 9
LETTER_W, GAP, SPACE = 10, 3, 12
NAME_W = 9 * LETTER_W + 8 * GAP + SPACE - GAP
NAME_C0 = (COLS - NAME_W) // 2


def name_cells():
    """(col, row) of every lit character cell of the name."""
    out, x = [], NAME_C0
    for ch in NAME:
        if ch == " ":
            x += SPACE - GAP
            continue
        for r, line in enumerate(PIXELS[ch]):
            for c, p in enumerate(line):
                if p == "#":
                    out += [(x + 2 * c, NAME_ROW + r), (x + 2 * c + 1, NAME_ROW + r)]
        x += LETTER_W + GAP
    return out


def name_mask():
    """Letters as a boolean wall mask on the simulation grid."""
    m = np.zeros((NY, NX), bool)
    for c, r in name_cells():
        m[r * SY:(r + 1) * SY, c * SX:(c + 1) * SX] = True
    return m


# ---------------------------------------------------------------- waves
def laplacian(u):
    return np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u


class Tank:
    def __init__(self, walls):
        p = PADDING
        self.shape = (NY + 2 * p, NX + 2 * p)
        self.walls = np.zeros(self.shape, bool)
        self.walls[p:-p, p:-p] = walls
        yy, xx = np.mgrid[:self.shape[0], :self.shape[1]]
        edge = np.minimum.reduce([yy, xx, self.shape[0] - 1 - yy, self.shape[1] - 1 - xx])
        self.sigma = 0.12 * np.clip(1 - edge / p, 0, 1) ** 2
        self.u = np.zeros(self.shape, np.float32)
        self.prev = np.zeros(self.shape, np.float32)
        self.yy, self.xx = yy, xx

    def blob(self, x, y, r=2.5):
        d2 = (self.xx - PADDING - x) ** 2 + (self.yy - PADDING - y) ** 2
        return np.exp(-d2 / (2 * r * r)).astype(np.float32)

    def step(self, drive=None, damp=0.0):
        s = self.sigma + damp
        nxt = (2 * self.u - (1 - s) * self.prev + C2 * laplacian(self.u)) / (1 + s)
        if drive is not None:
            nxt += drive
        nxt[self.walls] = 0
        self.prev, self.u = self.u, nxt

    def view(self):
        p = PADDING
        return self.u[p:-p, p:-p].copy()


def hum(walls, x, y):
    """Steady state of a continuous source: exactly one period of frames."""
    tank = Tank(walls)
    b = tank.blob(x, y)
    steps = PERIOD_FRAMES * STEPS_PER_FRAME
    w = 2 * math.pi / steps
    frames = []
    for n in range(steps * 70):
        if n >= steps * 69 and n % STEPS_PER_FRAME == 0:
            frames.append(tank.view())
        tank.step(b * (0.25 * math.sin(w * n)), damp=0.0008)
    drift = np.abs(tank.view() - frames[0]).max() / np.abs(frames[0]).max()
    print(f"hum: repeats to within {drift:.4f}")
    return np.array(frames)


def burst(walls, x, y):
    """A short off-frequency burst, LOOP_FRAMES long, silent before BURST_AT."""
    tank = Tank(walls)
    b = tank.blob(x, y, 2.0)
    w = 2 * math.pi * 0.5 / BURST_WAVELENGTH
    n_on = int(3.5 * BURST_WAVELENGTH / 0.5)
    frames, n = [], 0
    for f in range(LOOP_FRAMES):
        frames.append(tank.view())
        if f < BURST_AT:
            continue
        damp = 0.002 + 0.05 * min(max((f - 84) / 26, 0), 1) ** 2      # let it ring, then calm the water
        for _ in range(STEPS_PER_FRAME):
            drive = b * (0.3 * math.sin(w * n) * math.sin(math.pi * n / n_on) ** 2) if n < n_on else None
            tank.step(drive, damp=damp)
            n += 1
    frames = np.array(frames)
    print(f"burst: residual at loop end {np.abs(frames[-1]).max() / np.abs(frames).max():.4f}")
    return frames


def analytic(frames):
    """Real frames (t, y, x) of a looping narrow-band field -> complex analytic signal."""
    spec = np.fft.fft(frames, axis=0)
    n = len(frames)
    spec[n // 2:] = 0
    spec[1:n // 2] *= 2
    return np.fft.ifft(spec, axis=0).astype(np.complex64)


# ---------------------------------------------------------------- crests as line art
def crest_glyphs(z, blocked):
    """One complex frame (NY, NX) -> (chars, strength) on the cell grid.

    A crest is where the phase of z is zero. Each cell it runs through gets
    a glyph by slope; gentle slopes use _ and - at half-row steps.
    """
    amp = np.abs(z) + 1e-9
    gy, gx = np.gradient(z)
    gx = (np.conj(z) * gx).imag / amp ** 2          # phase gradient
    gy = (np.conj(z) * gy).imag / amp ** 2
    g = np.hypot(gx, gy)
    line = (np.abs(np.angle(z)) < 0.56 * g) & (g < 1.2) & ~blocked

    def cells(a):
        return a.reshape(ROWS, SY, COLS, SX).sum((1, 3))

    n = cells(line.astype(np.float32))
    nz = np.maximum(n, 1)
    yy = np.arange(NY)[:, None] % SY + 0.5
    cy = cells(line * yy) / nz / SY                  # 0 top .. 1 bottom
    sx, sy = cells(line * np.abs(gx)) / nz, cells(line * np.abs(gy)) / nz
    sign = cells(line * gx * gy) / nz                # > 0: crest rises to the right
    beta = np.degrees(np.arctan2(sx, sy))            # 0 = horizontal crest, 90 = vertical

    flat, steep = (beta < 42) & (n >= 2), (beta > 74) & (n >= 3)
    diag = ~flat & ~steep & (n >= 3)
    # thin double strokes: keep the cell that holds more of the line
    left, right = np.roll(n, 1, 1), np.roll(n, -1, 1)
    steep &= ~(np.roll(steep, 1, 1) & (left >= n)) & ~(np.roll(steep, -1, 1) & (right > n))
    diag &= ~(np.roll(diag, 1, 1) & (left >= n)) & ~(np.roll(diag, -1, 1) & (right > n))
    up, down = np.roll(n, 1, 0), np.roll(n, -1, 0)
    flat &= ~(np.roll(flat, 1, 0) & (up >= n)) & ~(np.roll(flat, -1, 0) & (down > n))

    ch = np.full((ROWS, COLS), " ", dtype="<U1")
    ch[steep] = "|"
    ch[diag & (sign > 0)] = "/"
    ch[diag & (sign <= 0)] = "\\"
    ch[flat & (cy >= 0.36)] = "-"
    ch[flat & (cy >= 0.7)] = "_"
    high = flat & (cy < 0.36)                        # top of this cell = floor of the one above
    above = np.roll(high, -1, 0)
    above[-1] = False
    ch[above & (ch == " ")] = "_"
    return ch, cell_mean(amp)


# ---------------------------------------------------------------- composition
GAP_X = (NAME_C0 + 4 * LETTER_W + 3 * GAP + SPACE / 2) * SX      # middle of the space between the words
MID_Y = (NAME_ROW + 4.5) * SY
BURST_C, BURST_R = 111, 4                                         # cell of the off-frequency source
ROLES = "E M B E D D E D   S O F T W A R E     ·     M A C H I N E   L E A R N I N G"


def tone(strength, edges, names):
    cl = np.full(strength.shape, "", dtype="<U2")
    for e, n in zip(edges, names):
        cl[strength >= e] = n
    return cl


def label(g, c, r, s, cls, pad=1):
    g.text(c, r, s, cls, clear=False)
    g.cover(c - pad, r, c + len(s) - 1 + pad, r)


def build():
    walls = name_mask()
    wall_c = cell_mean(walls.astype(np.float32)) > 0

    # the hum
    Z = analytic(hum(walls, GAP_X, MID_Y))
    ref = np.percentile(np.abs(Z[0])[~walls], 97)
    hum_frames = []
    for z in Z:
        ch, amp = crest_glyphs(z, walls)
        cl = tone(amp / ref, [0.012, 0.06, 0.2, 0.75], ["t1", "t2", "t3", "t4"])
        ch[(cl == "") | wall_c] = " "
        g = Grid(COLS, ROWS)
        g.paste(ch, cl)
        hum_frames.append(g)

    # the burst
    B = analytic(burst(walls, (BURST_C + 0.5) * SX, (BURST_R + 0.5) * SY))
    bref = np.abs(B).max()
    energy = np.abs(B).mean((1, 2))
    energy /= energy.max()
    burst_frames = []
    for f, z in enumerate(B):
        g = Grid(COLS, ROWS)
        if f >= BURST_AT:
            ch, amp = crest_glyphs(z, walls)
            cl = tone(amp / bref, [0.03, 0.085], ["b", "a"])
            ch[(cl == "") | wall_c] = " "
            g.paste(ch, cl)
        burst_frames.append(g)

    # the name
    static = Grid(COLS, ROWS)
    for c, r in name_cells():
        static.put(c, r, "#", "t5")

    # instrument text: fixed labels plus a readout that follows the burst
    # (it updates four times a second, so it is its own, slower sprite)
    rng = np.random.default_rng(7)
    hud_frames = []
    for f in range(0, LOOP_FRAMES, FPS // HUD_FPS):
        g = Grid(COLS, ROWS)
        g.text(2, 0, "ripple tank", "t4", clear=False)
        g.text(15, 0, "560 x 224 fdtd  ·  hum 0.67 Hz  ·  one period, looped", "t2", clear=False)
        g.ctext(COLS // 2, ROWS - 1, ROLES, "t4", clear=False)
        err = 0.03 + 0.03 * rng.random() + 0.9 * energy[f]
        hot = err > 0.2
        right = COLS - 3
        status = "[ ANOMALY ]" if hot else "[ nominal ]"
        g.rtext(right, 0, status, "a" if hot else "t3", clear=False)
        g.rtext(right - len(status) - 2, 0, f"recon_err {err:.2f}", "a" if hot else "t3", clear=False)
        g.cover(0, 0, COLS - 1, 0)
        g.cover(0, ROWS - 1, COLS - 1, ROWS - 1)
        if hot:
            tag = "off-frequency source  0.95 Hz"
            g.put(BURST_C, BURST_R, "x", "a")
            label(g, BURST_C - len(tag) // 2, BURST_R - 2, tag, "a")
        hud_frames.append(g)

    movie = Movie(COLS, ROWS, pad=(12, 14), title="Arda Çabuk — embedded software and machine learning",
                  desc="An ASCII ripple tank. The name ARDA ÇABUK stands in the water as walls. A steady source "
                       "between the two words sends out rings that reflect off the letters and bend through the "
                       "gaps. Every ten seconds a second source fires at a different frequency; its rings are "
                       "drawn in orange and the readout switches from nominal to anomaly.")
    movie.add(hum_frames, FPS)
    movie.add(burst_frames, FPS, opaque=True)
    movie.add(static)
    movie.add(hud_frames, HUD_FPS, opaque=True)
    return movie


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("hero.png", times=(0, 3.0, 5.0, 7.5))
        m.preview("hero-light.png", times=(5.0,), theme="light")
    m.write("hero.svg")
