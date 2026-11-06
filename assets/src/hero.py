"""Hero banner: a field of noise is denoised, step by step, into a dot-matrix name.

The cedilla of the Ç is the one orange detail, the "anomaly" in the signal.
Loop: noise -> name -> hold -> back to noise (15 s).
"""
import math
import random

from kit import ACCENT, BG, DIM, EDGE, FAINT, INK, MONO, MUTED, SANS, f, write

random.seed(1907)

W, H = 1200, 460
T = 15  # seconds per loop
CELL = 16

GLYPHS = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "D": ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    " ": ["....."] * 7,
}
CEDILLA = ["..#..", ".##.."]  # rows 7-8 under the C

name = "ARDA CABUK"
widths = [2 if ch == " " else 5 for ch in name]
starts = [sum(w + 1 for w in widths[:i]) for i in range(len(name))]
cols = starts[-1] + widths[-1]
rows = 9
x0 = (W - (cols - 1) * CELL) / 2
y0 = 108


def pos(c, r):
    return x0 + c * CELL, y0 + r * CELL


targets = []  # (x, y, col, accent)
for i, ch in enumerate(name):
    g = GLYPHS[ch]
    for r, line in enumerate(g):
        for c, px in enumerate(line):
            if px == "#":
                targets.append((*pos(starts[i] + c, r), starts[i] + c, False))
    if i == 5:  # the C of ÇABUK gets its cedilla
        for r, line in enumerate(CEDILLA):
            for c, px in enumerate(line):
                if px == "#":
                    targets.append((*pos(starts[i] + c, 7 + r), starts[i] + c, True))


def noise_point():
    while True:
        x = random.gauss(W / 2, 330)
        y = random.gauss(180, 95)
        if 36 < x < W - 36 and 58 < y < 300:
            return x, y


def kt(*ts):
    return ";".join(f(t, 4) for t in ts)


SPL = "0.45 0 0.55 1;0.25 0.1 0.15 1;0 0 1 1;0.6 0 0.9 0.45;0.1 0.55 0.5 1"

out = []

# LED matrix in its "off" state
grid = []
for c in range(cols):
    for r in range(rows):
        x, y = pos(c, r)
        grid.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="1.3"/>')
out.append(f'<g fill="{FAINT}">{"".join(grid)}</g>')

# name dots
dots = []
for x, y, c, accent in targets:
    s = (c / (cols - 1)) * 0.07 + random.random() * 0.02
    a, b = 0.11 + s, 0.28 + s
    h = 0.80 + s * 0.5
    d = h + 0.10
    p0, p2 = noise_point(), noise_point()
    p1 = (p0[0] + random.uniform(-22, 22), p0[1] + random.uniform(-14, 14))
    tr = lambda p: f"{f(p[0] - x)} {f(p[1] - y)}"
    times = kt(0, a, b, h, d, 1)
    color = ACCENT if accent else INK
    dots.append(
        f'<circle cx="{f(x)}" cy="{f(y)}" r="4.7" fill="{color}">'
        f'<animateTransform attributeName="transform" type="translate" dur="{T}s" repeatCount="indefinite" '
        f'calcMode="spline" keyTimes="{times}" keySplines="{SPL}" values="{tr(p0)};{tr(p1)};0 0;0 0;{tr(p2)};{tr(p0)}"/>'
        f'<animate attributeName="r" dur="{T}s" repeatCount="indefinite" calcMode="spline" '
        f'keyTimes="{times}" keySplines="{SPL}" values="1.7;1.7;4.7;4.7;1.7;1.7"/>'
        f'<animate attributeName="fill" dur="{T}s" repeatCount="indefinite" calcMode="spline" '
        f'keyTimes="{times}" keySplines="{SPL}" values="{MUTED};{MUTED};{color};{color};{MUTED};{MUTED}"/>'
        f"</circle>"
    )
out.append(f"<g>{''.join(dots)}</g>")

# extra noise that is "removed" while the name forms
noise = []
for _ in range(170):
    q0, q2 = noise_point(), noise_point()
    q1 = (q0[0] + random.uniform(-22, 22), q0[1] + random.uniform(-14, 14))
    q3 = (W / 2 + (q1[0] - W / 2) * 1.12, 180 + (q1[1] - 180) * 1.25)
    s = random.random() * 0.08
    a, b = 0.11 + s, 0.25 + s
    h = 0.80 + s * 0.5
    d = h + 0.10
    times = kt(0, a, b, h, d, 1)
    rr = random.choice((1.2, 1.5, 1.8))
    noise.append(
        f'<circle cx="{f(q0[0])}" cy="{f(q0[1])}" r="{rr}">'
        f'<animateTransform attributeName="transform" type="translate" dur="{T}s" repeatCount="indefinite" '
        f'calcMode="spline" keyTimes="{times}" keySplines="{SPL}" '
        f'values="0 0;{f(q1[0] - q0[0])} {f(q1[1] - q0[1])};{f(q3[0] - q0[0])} {f(q3[1] - q0[1])};'
        f'{f(q3[0] - q0[0])} {f(q3[1] - q0[1])};{f(q2[0] - q0[0])} {f(q2[1] - q0[1])};0 0"/>'
        f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" calcMode="spline" '
        f'keyTimes="{times}" keySplines="{SPL}" values="0.7;0.7;0;0;0.7;0.7"/>'
        f"</circle>"
    )
out.append(f'<g fill="{MUTED}">{"".join(noise)}</g>')

# denoising step counter: t = 1000 -> 0 while the name forms, 0 -> 1000 while it dissolves
rev = [0.11 + k * 0.025 for k in range(11)]
fwd = [0.80 + j * 0.014 for j in range(11)]
counter = []
for v in range(0, 1001, 100):
    k = (1000 - v) // 100  # index in the reverse pass
    j = v // 100  # index in the forward pass
    spans = []
    r_start = 0 if k == 0 else rev[k]
    r_end = rev[k + 1] if k < 10 else fwd[0]
    spans.append((r_start, r_end))
    f_start = fwd[j] if j > 0 else None
    f_end = fwd[j + 1] if j < 10 else 1.0
    if f_start is not None:
        spans.append((f_start, f_end))
    elif v == 0:
        spans[-1] = (r_start, fwd[1])
    keys, vals = [0.0], [0]
    for s0, s1 in spans:
        if s0 == 0:
            vals[0] = 1
        else:
            keys.append(s0)
            vals.append(1)
        if s1 < 1:
            keys.append(s1)
            vals.append(0)
    counter.append(
        f'<text x="164" y="424" font-size="12" fill="{MUTED}" opacity="{vals[0]}">{v}'
        f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" calcMode="discrete" '
        f'keyTimes="{kt(*keys)}" values="{";".join(map(str, vals))}"/></text>'
    )
out.append(
    f'<text x="48" y="424" font-size="12" fill="{DIM}">denoising  t =</text>' + "".join(counter)
)

# tagline, revealed once the name has settled
reveal = kt(0, 0.34, 0.39, 0.78, 0.81, 1)
out.append(
    f'<g>'
    f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" calcMode="spline" '
    f'keyTimes="{reveal}" keySplines="0 0 1 1;0.3 0 0.2 1;0 0 1 1;0.5 0 1 1;0 0 1 1" values="0;0;1;1;0;0"/>'
    f'<animateTransform attributeName="transform" type="translate" dur="{T}s" repeatCount="indefinite" calcMode="spline" '
    f'keyTimes="{reveal}" keySplines="0 0 1 1;0.3 0 0.2 1;0 0 1 1;0.5 0 1 1;0 0 1 1" values="0 10;0 10;0 0;0 0;0 -6;0 10"/>'
    f'<text x="{W / 2}" y="322" style="font-family:{SANS};letter-spacing:-0.2px" font-size="28" fill="{INK}" text-anchor="middle">'
    f"Code close to the hardware. Models that notice when something is off.</text>"
    f'<text x="{W / 2}" y="358" font-size="13.5" fill="{MUTED}" text-anchor="middle" letter-spacing="0.5">'
    f"bare-metal C  ·  modern C++  ·  embedded Linux  ·  signal processing  ·  machine learning</text>"
    f"</g>"
)

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Arda Çabuk</title>
<desc id="d">A field of random dots is denoised into the name Arda Çabuk on a dot-matrix grid; the cedilla is orange. Tagline: Code close to the hardware. Models that notice when something is off.</desc>
<defs><clipPath id="frame"><rect width="{W}" height="{H}" rx="18"/></clipPath></defs>
<style>text {{ font-family: {MONO}; }}</style>
<g clip-path="url(#frame)">
<rect width="{W}" height="{H}" fill="{BG}"/>
<text x="48" y="44" font-size="12" fill="{DIM}">fig. 1 — a name, recovered from noise</text>
<text x="{W - 48}" y="44" font-size="12" fill="{DIM}" text-anchor="end">Eskişehir · 39.78°N 30.52°E</text>
<text x="{W - 48}" y="424" font-size="12" fill="{DIM}" text-anchor="end">embedded × machine learning</text>
{"".join(out)}
</g>
<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{EDGE}"/>
</svg>
"""

write("hero.svg", svg)
