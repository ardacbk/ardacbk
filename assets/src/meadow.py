"""ASCII hero: a night meadow at a wind test site.

Grass bends under gusts that roll across the field, two turbines turn on the
hill (the big one has a blinking aviation light), fireflies drift, stars
twinkle, a shooting star crosses once per loop, and a terminal at the bottom
keeps monitoring the turbine. A signpost in the grass carries the name.

Every frame is a real character grid. Cells that never change go into one
static layer; the rest are drawn per frame and switched with SMIL, so the SVG
stays small and plays through a plain <img> on GitHub.
"""
import math
import random

from kit import ACCENT, BG, EDGE, INK, MONO, MUTED, write

W, H = 1200, 460
COLS, ROWS = 112, 24
CW, RH = 10.4, 18.0
X0 = (W - COLS * CW) / 2
Y0 = 12
FPS, F = 8, 48  # 6 s loop
TAU = 2 * math.pi
NBSP = " "

STYLE = {
    "star": "#d8d2c4", "stard": "#7d776c", "moon": "#e9dcb5", "hill": "#46423b",
    "tower": "#9a958b", "blade": INK, "light": ACCENT, "wood": "#a8835a",
    "sign": INK, "name": INK, "grassb": "#4c5737", "grass": "#7d8f55",
    "grasst": "#a2b36c", "poppy": ACCENT, "fly": "#f2d675", "flyd": "#8f7d3c",
    "prompt": "#c9b48a", "muted": MUTED, "ink": INK,
}
PRIORITY = ["stard", "star", "moon", "hill", "tower", "blade", "light", "wood", "sign", "name",
            "grassb", "grass", "grasst", "poppy", "flyd", "fly", "prompt", "muted", "ink"]

rng = random.Random(1907)


def blank():
    return [[None] * COLS for _ in range(ROWS)]


def put(g, c, r, ch, cls):
    if 0 <= r < ROWS and 0 <= c < COLS and ch != " ":
        cur = g[r][c]
        if cur is None or PRIORITY.index(cls) >= PRIORITY.index(cur[1]):
            g[r][c] = (ch, cls)


def text(g, c, r, s, cls):
    for i, ch in enumerate(s):
        put(g, c + i, r, ch, cls)


# ---------------------------------------------------------------- static parts
def draw_moon(g):
    for r, line in enumerate(["  .--.", " /  .'", "|  (", " \\  '.", "  '--'"]):
        text(g, 5, 1 + r, line, "moon")


def hill_y(c):
    return (16.2 - 4.4 * math.exp(-((c - 86) / 13) ** 2) - 3.0 * math.exp(-((c - 105) / 7) ** 2)
            - 1.6 * math.exp(-((c - 62) / 8) ** 2))


def draw_hills(g):
    # one character per column; its glyph height inside the cell follows the
    # curve: ' near the top, - in the middle, . and _ near the bottom
    for c in range(46, COLS):
        y = hill_y(c)
        r, frac = int(y), y - int(y)
        ch = "'" if frac < 0.3 else "-" if frac < 0.6 else "." if frac < 0.85 else "_"
        put(g, c, r, ch, "hill")
    for c in range(46, COLS):  # faint texture on the slope
        if rng.random() < 0.1:
            put(g, c, int(hill_y(c)) + 1 + rng.randrange(2), rng.choice(".'`"), "hill")


SIGN_POST = 28  # left column of the post "|  |"


def draw_sign(g):
    p = SIGN_POST
    name = "|   A R D A   Ç A B U K   |"
    left = p + 2 - len(name) // 2
    top = "." + "-" * (len(name) - 2) + "."
    text(g, left, 5, top, "wood")
    text(g, left, 6, "|", "wood")
    text(g, left + 1, 6, name[1:-1], "name")
    text(g, left + len(name) - 1, 6, "|", "wood")
    bottom = "'" + "-" * (p - left - 1) + ".  ." + "-" * (left + len(name) - p - 6) + "'"
    text(g, left, 7, bottom, "wood")
    for r in range(8, 21):
        text(g, p, r, "|  |", "wood")
    # left plank: embedded
    lp = "embedded · C/C++"
    lw = len(lp) + 6
    text(g, p - lw, 9, "." + "-" * (lw - 1), "wood")
    text(g, p - lw - 1, 10, "<", "wood")
    text(g, p - lw + 2, 10, lp, "sign")
    text(g, p - lw, 11, "'" + "-" * (lw - 1), "wood")
    # right plank: machine learning
    rp = "machine learning"
    rw = len(rp) + 6
    text(g, p + 4, 12, "-" * (rw - 1) + ".", "wood")
    text(g, p + 7, 13, rp, "sign")
    text(g, p + 4 + rw, 13, ">", "wood")
    text(g, p + 4, 14, "-" * (rw - 1) + "'", "wood")


TURBINES = [  # hub col, hub row, radius (rows), frames per turn, phase, tower bottom row
    (86, 5, 4.6, 24, 0.0, int(hill_y(86)) - 1, True),
    (105, 9, 2.6, 16, 40.0, int(hill_y(105)) - 1, False),
]


def draw_towers(g):
    for cx, cy, R, _, _, bottom, _ in TURBINES:
        for r in range(cy + 1, bottom + 1):
            put(g, cx, r, "|", "tower")


# ---------------------------------------------------------------- animated parts
ASPECT = RH / CW  # columns per row that look the same length on screen


def blade_char(th):
    a = math.degrees(math.atan2(math.sin(th) * RH, math.cos(th) * ASPECT * CW)) % 180
    return "-" if a < 22.5 or a >= 157.5 else "/" if a < 67.5 else "|" if a < 112.5 else "\\"


def draw_rotors(g, f):
    for cx, cy, R, per, ph, _, light in TURBINES:
        base = math.radians(ph) - TAU * f / per  # clockwise
        for b in range(3):
            th = base + b * TAU / 3
            dx, dy = R * math.cos(th) * ASPECT, -R * math.sin(th)
            a = math.degrees(math.atan2(abs(math.sin(th)), abs(math.cos(th))))  # on-screen tilt
            if a < 33:  # near-horizontal: one cell per column, glyph height tracks the line
                n = max(1, round(abs(dx)))
                for i in range(1, n + 1):
                    yc = cy + 0.5 + dy * i / n
                    r, frac = math.floor(yc), yc - math.floor(yc)
                    ch = "'" if frac < 0.3 else "-" if frac < 0.62 else "." if frac < 0.85 else "_"
                    put(g, round(cx + dx * i / n), r, ch, "blade")
            else:  # one cell per row; "/" or "\\" when tilted, "|" when near-vertical
                n = max(1, round(abs(dy)))
                slash = "/" if dx * dy < 0 else "\\"
                pc = cx
                for j in range(1, n + 1):
                    r, c = round(cy + dy * j / n), round(cx + dx * j / n)
                    ch = ("|" if c == pc else slash) if a > 70 else slash
                    put(g, c, r, ch, "blade")
                    if a <= 70 and abs(dx / n) > 1.45:  # fill the gap on shallow diagonals
                        put(g, round(cx + dx * (j - 0.5) / n), r, slash, "blade")
                    pc = c
        on = light and (f % 8) < 3
        put(g, cx, cy, "@" if on else "o", "light" if on else "blade")


STARS = []
while len(STARS) < 64:
    c, r = rng.randrange(1, COLS - 1), rng.randrange(0, 12)
    if (3 <= c <= 13 and r <= 6) or (4 <= c <= 50 and 4 <= r <= 15) or (74 <= c <= 98 and r <= 10):
        continue
    STARS.append((c, r, rng.choice((8, 12, 16, 24, 48)), rng.randrange(48), rng.random() < 0.3))


def draw_stars(g, f):
    for c, r, per, ph, dim in STARS:
        k = (f + ph) % per
        if dim:
            put(g, c, r, "." if k != 0 else "'", "stard")
        else:
            put(g, c, r, {0: "*", 1: "+", per - 1: "+"}.get(k, "."), "star")


def draw_shooting_star(g, f):
    k = f - 14
    if 0 <= k < 9:
        for t, ch in ((0, "*"), (1, "-"), (2, "-"), (3, ".")):
            if k - t >= 0:
                put(g, 52 + 3 * (k - t), 1 + (k - t) // 2, ch, "star" if t < 2 else "stard")


def wind(c, f):
    gust = (0.5 + 0.5 * math.sin(TAU * (c / 38 - 2 * f / F))) ** 3
    flutter = 0.14 * math.sin(TAU * (3 * f / F) + c * 0.9)
    return 0.15 + 1.25 * gust + flutter


BLADES = []
for c in range(COLS):
    if rng.random() < 0.78:
        h = rng.choices((1, 2, 3, 4), (3, 4, 3, 2))[0]
        BLADES.append((c, 20, h, rng.choice((-0.5, 0, 0, 0.5)), rng.random() * TAU,
                       "poppy" if h >= 3 and rng.random() < 0.08 else "front", rng.choice("\",'|")))
    if rng.random() < 0.45:
        BLADES.append((c, rng.choice((17, 18)), rng.choice((1, 1, 2)), rng.choice((-0.5, 0, 0.5)),
                       rng.random() * TAU, "back", rng.choice("\",'")))


TUFTS = [c for c in range(2, COLS - 2, 1) if rng.random() < 0.09]


def draw_grass(g, f):
    for c, base, h, tilt, ph, kind, tuft in BLADES:
        lean = wind(c, f) + tilt + 0.25 * math.sin(TAU * 2 * f / F + ph)
        cls = "grassb" if kind == "back" else "grass"
        if h == 1:  # tiny tuft: only a strong gust flattens it
            put(g, c, base, "/" if lean > 1.0 else tuft, cls)
            continue
        prev = 0.0
        for k in range(h):
            x = lean * (k / (h - 1)) ** 1.6 * (h - 1) * 0.85
            d = x - prev
            ch = "/" if d > 0.35 else "\\" if d < -0.35 else "|"
            if k == h - 1:
                cls = "grasst" if kind != "back" else cls
                if kind == "poppy":
                    ch, cls = "*", "poppy"
            put(g, c + round(x), base - k, ch, cls)
            prev = x
    for c in TUFTS:  # little fans of grass that only shiver
        lean = wind(c, f)
        put(g, c - 1, 20, "\\", "grass")
        put(g, c, 19 if lean < 1.0 else 20, "|" if lean < 1.0 else "/", "grass")
        put(g, c + 1, 20, "/", "grass")
    for c in range(0, COLS, 3):
        put(g, c + (c * 7) % 3, 21, "'`,."[(c * 5) % 4], "grassb")


FLIES = [(rng.uniform(4, 70), rng.uniform(11, 19), rng.uniform(2, 7), rng.uniform(0.5, 1.6),
          rng.choice((1, 2)), rng.choice((1, 2)), rng.random() * TAU, rng.random() * TAU, rng.choice((2, 3)))
         for _ in range(9)]


def draw_flies(g, f):
    for c0, r0, ax, ay, m, n, p1, p2, bl in FLIES:
        c = round(c0 + ax * math.sin(TAU * m * f / F + p1))
        r = round(r0 + ay * math.sin(TAU * n * f / F + p2))
        glow = math.sin(TAU * bl * f / F + p1 * 2)
        if 0 <= r < ROWS and 0 <= c < COLS and g[r][c] and g[r][c][1] in ("wood", "sign", "name"):
            continue
        if glow > 0.35:
            put(g, c, r, "*", "fly")
        elif glow > -0.4:
            put(g, c, r, ".", "flyd")


def draw_status(g, f):
    mean = max(0.0, wind(TURBINES[0][0], f))  # anemometer on the big turbine
    text(g, 1, 23, "arda@eskisehir", "prompt")
    text(g, 15, 23, ":~$", "muted")
    text(g, 19, 23, "./monitor --site meadow", "ink")
    if f % 8 < 5:
        text(g, 43, 23, "_", "ink")
    stats = [("wind ", f"{3.4 + 2.4 * mean:.1f} m/s"), ("rotor ", "20 rpm"),
             ("vib[0..27] ", "ok"), ("recon_err ", f"{0.06 + 0.05 * mean:.2f}")]
    line = "   ".join(a + b for a, b in stats)
    c = COLS - 1 - len(line)
    for a, b in stats:
        text(g, c, 23, a, "muted")
        text(g, c + len(a), 23, b, "ink")
        c += len(a) + len(b) + 3


def frame(f):
    g = blank()
    draw_stars(g, f)
    draw_shooting_star(g, f)
    draw_moon(g)
    draw_hills(g)
    draw_towers(g)
    draw_rotors(g, f)
    draw_sign(g)
    draw_grass(g, f)
    draw_flies(g, f)
    draw_status(g, f)
    return g


# ---------------------------------------------------------------- SVG output
def esc(ch):
    return {"<": "&lt;", ">": "&gt;", "&": "&amp;", " ": NBSP}.get(ch, ch)


def runs(cells):
    """cells: {(row, col): (ch, cls)} -> <g class> blocks of <text> runs."""
    by = {}
    for (r, c), (ch, cls) in cells.items():
        by.setdefault(cls, {}).setdefault(r, []).append((c, ch))
    out = []
    for cls in PRIORITY:
        if cls not in by:
            continue
        texts = []
        for r, items in sorted(by[cls].items()):
            items.sort()
            group = [items[0]]
            for it in items[1:] + [None]:
                if it is not None and it[0] - group[-1][0] <= 10:
                    group.append(it)
                    continue
                c0, c1 = group[0][0], group[-1][0]
                chars = [NBSP] * (c1 - c0 + 1)
                for c, ch in group:
                    chars[c - c0] = esc(ch)
                texts.append(f'<text x="{X0 + c0 * CW:.1f}" y="{Y0 + (r + 1) * RH - 4.5:.1f}" '
                             f'textLength="{len(chars) * CW:.1f}">{"".join(chars)}</text>')
                group = [it] if it else []
        out.append(f'<g class="{cls}">{"".join(texts)}</g>')
    return "".join(out)


frames = [frame(f) for f in range(F)]
static, dynamic = {}, [dict() for _ in range(F)]
for r in range(ROWS):
    for c in range(COLS):
        vals = [fr[r][c] for fr in frames]
        if all(v == vals[0] for v in vals):
            if vals[0]:
                static[(r, c)] = vals[0]
        else:
            for f, v in enumerate(vals):
                if v:
                    dynamic[f][(r, c)] = v

dur = F / FPS
layers = []
for f in range(F):
    a, b = f / F, (f + 1) / F
    if f == 0:
        kt, vals = f"0;{b:.4f}", "1;0"
    else:
        kt, vals = f"0;{a:.4f};{b:.4f}" if b < 1 else f"0;{a:.4f}", "0;1;0" if b < 1 else "0;1"
    layers.append(f'<g opacity="{1 if f == 0 else 0}"><animate attributeName="opacity" dur="{dur}s" '
                  f'repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{vals}"/>'
                  f"{runs(dynamic[f])}</g>")

css = "".join(f".{k}{{fill:{v}}}" for k, v in STYLE.items())
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Arda Çabuk</title>
<desc id="d">An ASCII night meadow: grass bends in gusts of wind, two wind turbines turn on a hill, fireflies drift and stars twinkle. A signpost in the grass reads ARDA ÇABUK, pointing left to embedded and C/C++ and right to machine learning. A terminal line at the bottom monitors the turbine.</desc>
<style>text{{font-family:{MONO};font-size:17px;white-space:pre}}.name{{font-weight:700}}{css}</style>
<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>
<line x1="{X0:.1f}" y1="{Y0 + 22 * RH + 4:.1f}" x2="{X0 + COLS * CW:.1f}" y2="{Y0 + 22 * RH + 4:.1f}" stroke="{EDGE}"/>
<g>
{runs(static)}
{"".join(layers)}
</g>
<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{EDGE}"/>
</svg>
"""
write("hero.svg", svg)
print(f"  static cells: {len(static)}, dynamic cells/frame: {sum(map(len, dynamic)) / F:.0f}")
