"""Hero: a room at three in the morning, ray-marched and printed in ASCII.

The room is a real 3-D scene made of signed distance functions: walls with an
arched window cut through the back one, a desk, a monitor, a lamp, a shelf of
books, a plant, a cat on the sill and someone at the keyboard. It is lit by
three lights: the moon through the window (with its beam scattered in the
air), a warm desk lamp and the cold glow of the monitor.

Every character cell is 4 x 8 rays. The rays give a colour picture; the glyph
for a cell is the one whose shape best matches the brightness under it, and
its colour is the cell's colour snapped to a small palette.

The camera never moves, so the room is rendered once. What moves is redrawn
per frame on top: the sky through the window (stars, a shooting star, wind
turbines turning on the ridge with their red lights), the vibration trace on
the monitor, the board's blinking LED and the steam from the mug.
"""
import math
import pathlib
import sys

import numpy as np

from engine import SX, SY, Grid, Matcher, Movie, pack

COLS, ROWS = 200, 56
W, H = COLS * SX, ROWS * SY
FPS, FRAMES = 8, 64                    # an 8 s loop
SLOW = 4                               # twinkling updates every 4th frame

rng = np.random.default_rng(3)


# ================================================================ sdf toolkit
def V(*a):
    return np.array(a, float)


def length(p):
    return np.sqrt((p * p).sum(-1))


def box(p, c, h):
    q = np.abs(p - c) - h
    return length(np.maximum(q, 0)) + np.minimum(q.max(-1), 0)


def rbox(p, c, h, r):
    return box(p, c, np.asarray(h) - r) - r


def sphere(p, c, r):
    return length(p - c) - r


def capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip(pa @ ba / (ba @ ba), 0, 1)
    return length(pa - h[:, None] * ba) - r


def cyl_y(p, c, r, hh):
    d = np.stack([np.hypot(p[:, 0] - c[0], p[:, 2] - c[2]) - r, np.abs(p[:, 1] - c[1]) - hh], -1)
    return np.minimum(d.max(-1), 0) + length(np.maximum(d, 0))


def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = length(q)
    k1 = length(q / r)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-6)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


# ================================================================ the room
BACK_Z, WALL_T = 3.5, 0.3               # inner face of the back wall, thickness
WIN_Z = BACK_Z + WALL_T / 2              # plane of the window frame
WX0, WX1, WY0, SPRING = -0.25, 2.85, 0.8, 1.75
WCX, WR = (WX0 + WX1) / 2, (WX1 - WX0) / 2
MULL_X = (WX0 + (WX1 - WX0) / 3, WX0 + 2 * (WX1 - WX0) / 3)


def hole2d(x, y):
    """Window opening in the back wall (<0 inside): a rectangle under a half disc."""
    rect = np.maximum(np.maximum(WX0 - x, x - WX1), np.maximum(WY0 - y, y - SPRING))
    arch = np.maximum(np.hypot(x - WCX, y - SPRING) - WR, SPRING - y)
    return np.minimum(rect, arch)


def bars2d(x, y):
    """Window frame and glazing bars (<0 on a bar)."""
    edge = np.abs(hole2d(x, y) + 0.045) - 0.045
    ring = np.maximum(np.abs(np.hypot(x - WCX, y - SPRING) - 0.62) - 0.022, SPRING - y)
    bars = np.minimum.reduce([np.abs(x - MULL_X[0]) - 0.028, np.abs(x - MULL_X[1]) - 0.028,
                              np.abs(y - SPRING) - 0.028, np.abs(y - 1.18) - 0.02])
    return np.minimum.reduce([edge, ring, bars])


DESK_Y = 0.76
MON_C, MON_H = V(-1.42, 1.2, 3.22), V(0.56, 0.33, 0.03)
LAMP_BULB = V(-0.62, 1.3, 2.98)
MUG = V(-0.86, DESK_Y + 0.06, 2.83)
BOARD = V(-2.38, DESK_Y + 0.012, 2.95)

# fairy lights draped across the wall above the desk: two sagging spans
FAIRY = []
for _a, _b in ((-3.3, -1.9), (-1.9, -0.45)):
    for _s in np.linspace(0, 1, 9)[(0 if _a < -2 else 1):]:
        FAIRY.append(V(_a + (_b - _a) * _s, 3.17 - 0.15 * math.sin(math.pi * _s), BACK_Z - 0.03))
FAIRY_COL = [V(1.0, 0.72, 0.35), V(1.0, 0.45, 0.3), V(1.0, 0.85, 0.55), V(0.95, 0.55, 0.75)]

# books: (x0, width, height, colour)
BOOKS = []
_x = -2.95
while _x < -1.15:
    _w = rng.uniform(0.045, 0.085)
    _h = rng.uniform(0.19, 0.29)
    _c = rng.choice([[0.55, 0.16, 0.12], [0.16, 0.3, 0.45], [0.62, 0.48, 0.24], [0.2, 0.38, 0.28],
                     [0.5, 0.45, 0.4], [0.38, 0.2, 0.36], [0.72, 0.6, 0.42]])
    BOOKS.append((_x, _w, _h, np.array(_c) * rng.uniform(0.75, 1.1)))
    _x += _w + rng.uniform(0.003, 0.012)
    if rng.random() < 0.12:
        _x += 0.12
BOOK_COL = np.array([b[3] for b in BOOKS])


POSE = {"nod": 0.0, "tl": 0.0, "tr": 0.0, "tail": 0.0}   # the moving bits, set per frame


def parts(p):
    """Every object as (name, distance). Names double as materials."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    out = []
    # shell
    out.append(("floor", y))
    out.append(("wall", 3.45 - y))                                         # ceiling
    out.append(("wall", x + 3.35))
    out.append(("wall", 3.4 - x))
    out.append(("wall", np.maximum(np.abs(z - WIN_Z) - WALL_T / 2, -hole2d(x, y))))
    out.append(("frame", np.maximum(np.maximum(bars2d(x, y), hole2d(x, y) - 0.001), np.abs(z - WIN_Z) - 0.035)))
    out.append(("sill", box(p, V(WCX, WY0 - 0.035, BACK_Z - 0.05), V(WR + 0.12, 0.035, 0.2))))
    out.append(("trim", box(p, V(0, 0.05, BACK_Z - 0.012), V(4, 0.05, 0.012))))     # skirting
    # desk
    d = box(p, V(-1.75, DESK_Y - 0.02, 3.1), V(1.25, 0.022, 0.4))
    for lx in (-2.9, -0.6):
        d = np.minimum(d, box(p, V(lx, DESK_Y / 2, 3.1), V(0.03, DESK_Y / 2, 0.36)))
    d = np.minimum(d, box(p, V(-0.82, 0.5, 3.1), V(0.2, 0.22, 0.36)))           # drawers
    out.append(("desk", d))
    # monitor
    out.append(("bezel", np.minimum.reduce([
        rbox(p, MON_C, MON_H, 0.015),
        box(p, V(MON_C[0], 0.95, 3.3), V(0.03, 0.15, 0.025)),
        box(p, V(MON_C[0], DESK_Y + 0.008, 3.28), V(0.16, 0.008, 0.1))])))
    out.append(("screen", box(p, V(MON_C[0], MON_C[1], MON_C[2] - 0.025),
                              V(MON_H[0] - 0.03, MON_H[1] - 0.03, 0.008))))
    out.append(("keys", rbox(p, V(-1.5, DESK_Y + 0.012, 2.92), V(0.24, 0.012, 0.08), 0.008)))
    # lamp
    base = V(-0.5, DESK_Y, 3.32)
    elbow = V(-0.48, 1.52, 3.36)
    head = V(-0.62, 1.42, 3.0)
    out.append(("lamp", np.minimum.reduce([
        cyl_y(p, base + V(0, 0.012, 0), 0.09, 0.012),
        capsule(p, base, elbow, 0.014),
        capsule(p, elbow, head, 0.014),
        np.maximum(sphere(p, head + V(0, -0.02, 0), 0.12),
                   np.maximum(y - (head[1] + 0.02), (head[1] - 0.09) - y)),
    ])))
    out.append(("bulb", sphere(p, LAMP_BULB, 0.035)))
    # mug, board
    out.append(("mug", np.minimum(cyl_y(p, MUG, 0.045, 0.06),
                                  capsule(p, MUG + V(0.05, 0.03, 0), MUG + V(0.05, -0.03, 0), 0.012))))
    out.append(("board", box(p, BOARD, V(0.12, 0.008, 0.07))))
    # shelf and books
    sh = box(p, V(-1.95, 1.9, 3.36), V(1.05, 0.018, 0.15))
    for bx in (-2.8, -1.1):
        sh = np.minimum(sh, box(p, V(bx, 1.84, 3.46), V(0.012, 0.06, 0.03)))
    out.append(("shelf", sh))
    books = np.full(len(p), 9.0)
    ids = np.zeros(len(p), int)
    for i, (bx, bw, bh, _) in enumerate(BOOKS):
        db = box(p, V(bx + bw / 2, 1.918 + bh / 2, 3.38), V(bw / 2, bh / 2, 0.11))
        ids = np.where(db < books, i, ids)
        books = np.minimum(books, db)
    out.append(("book", books))
    parts.book_ids = ids
    out.append(("pot", cyl_y(p, V(-1.0, 1.98, 3.38), 0.06, 0.06)))
    leaves = np.full(len(p), 9.0)
    for a in range(9):
        ang = a * 2.4
        tip = V(-1.0 + 0.2 * math.cos(ang), 2.0 - 0.15 - 0.06 * (a % 3), 3.38 + 0.12 * math.sin(ang))
        leaves = np.minimum(leaves, capsule(p, V(-1.0, 2.04, 3.38), tip, 0.022))
    out.append(("leaf", leaves))
    # chair
    out.append(("chair", np.minimum.reduce([
        rbox(p, V(-1.55, 0.5, 2.3), V(0.24, 0.035, 0.23), 0.02),
        rbox(p, V(-1.55, 0.78, 2.05), V(0.22, 0.22, 0.03), 0.03),
        capsule(p, V(-1.55, 0.05, 2.3), V(-1.55, 0.48, 2.3), 0.025),
        capsule(p, V(-1.85, 0.03, 2.3), V(-1.25, 0.03, 2.3), 0.02),
        capsule(p, V(-1.55, 0.03, 2.0), V(-1.55, 0.03, 2.6), 0.02)])))
    # the person, seen from behind, leaning in
    hip, chest = V(-1.55, 0.62, 2.22), V(-1.53, 1.08, 2.38)
    torso = capsule(p, hip, chest, 0.17)
    torso = smin(torso, capsule(p, V(-1.76, 1.06, 2.38), V(-1.3, 1.06, 2.4), 0.09), 0.08)
    ls, rs = V(-1.76, 1.05, 2.4), V(-1.3, 1.05, 2.42)
    tl, tr = POSE["tl"], POSE["tr"]
    le, re = V(-1.82, 0.84 + 0.015 * tl, 2.58), V(-1.22, 0.84 + 0.015 * tr, 2.6)
    lh, rh = V(-1.63, 0.8 + 0.07 * tl, 2.9 - 0.03 * tl), V(-1.38, 0.8 + 0.07 * tr, 2.9 - 0.03 * tr)
    arms = np.minimum.reduce([capsule(p, ls, le, 0.062), capsule(p, le, lh, 0.05),
                              capsule(p, rs, re, 0.062), capsule(p, re, rh, 0.05)])
    hood = ellipsoid(p, V(-1.53, 1.16, 2.33), V(0.15, 0.08, 0.1))
    out.append(("hoodie", smin(smin(torso, arms, 0.04), hood, 0.04)))
    out.append(("skin", np.minimum.reduce([sphere(p, lh, 0.04), sphere(p, rh, 0.04),
                                           capsule(p, V(-1.53, 1.12, 2.4), V(-1.52, 1.24 - 0.05 * POSE["nod"], 2.45 + 0.04 * POSE["nod"]), 0.05)])))
    headc = V(-1.52, 1.34 - 0.085 * POSE["nod"], 2.47 + 0.07 * POSE["nod"])
    out.append(("hair", ellipsoid(p, headc, V(0.105, 0.12, 0.115))))
    out.append(("phones", np.minimum.reduce([
        np.maximum(np.abs(np.hypot(x - headc[0], y - headc[1]) - 0.13) - 0.014,
                   np.maximum(np.abs(z - headc[2]) - 0.02, headc[1] - y)),
        ellipsoid(p, headc + V(-0.115, -0.01, 0), V(0.03, 0.055, 0.05)),
        ellipsoid(p, headc + V(0.115, -0.01, 0), V(0.03, 0.055, 0.05))])))
    # the cat on the sill, a silhouette against the sky
    cat = V(2.25, WY0, BACK_Z - 0.06)
    body = ellipsoid(p, cat + V(0, 0.13, 0), V(0.12, 0.14, 0.1))
    cathead = sphere(p, cat + V(0.02, 0.32, -0.02), 0.075)
    ears = np.minimum(capsule(p, cat + V(-0.02, 0.35, -0.02), cat + V(-0.035, 0.405, -0.02), 0.01),
                      capsule(p, cat + V(0.06, 0.35, -0.02), cat + V(0.075, 0.405, -0.02), 0.01))
    a = 0.7 + POSE["tail"]
    knee = cat + V(-0.26, 0.02, -0.04)
    tip = knee + 0.15 * V(-math.sin(a), math.cos(a), -0.25)
    tail = np.minimum(capsule(p, cat + V(-0.1, 0.03, 0.02), knee, 0.022), capsule(p, knee, tip, 0.019))
    out.append(("cat", smin(smin(body, cathead, 0.04), np.minimum(ears, tail), 0.02)))
    wire = np.full(len(p), 9.0)
    for a, b in zip(FAIRY[:-1], FAIRY[1:]):
        wire = np.minimum(wire, capsule(p, a, b, 0.004))
    out.append(("phones", wire))
    # rug
    out.append(("rug", rbox(p, V(-0.9, 0.006, 1.6), V(1.3, 0.006, 0.95), 0.004)))
    # pinboard on the right wall
    out.append(("cork", box(p, V(3.39, 1.55, 1.6), V(0.012, 0.42, 0.62))))
    return out


NAMES = [n for n, _ in parts(np.zeros((1, 3)))]


def sdf(p):
    return np.minimum.reduce([d for _, d in parts(p)])


def material(p):
    ds = parts(p)
    return np.argmin(np.stack([d for _, d in ds]), 0), parts.book_ids


# ================================================================ camera
CAM = V(0.45, 1.38, -0.35)
LOOK = V(-0.12, 1.4, 3.5)
HFOV = math.radians(80)


def camera_rays():
    f = LOOK - CAM
    f /= np.linalg.norm(f)
    r = np.cross(V(0, 1, 0), f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    tx = math.tan(HFOV / 2)
    xs = ((np.arange(W) + 0.5) / W * 2 - 1) * tx
    ys = -((np.arange(H) + 0.5) / H * 2 - 1) * tx * H / W
    X, Y = np.meshgrid(xs, ys)
    d = f[None] + X.reshape(-1, 1) * r[None] + Y.reshape(-1, 1) * u[None]
    return d / length(d)[:, None]


def march(o, d, tmax=14.0, steps=220):
    n = len(d)
    t = np.full(n, 0.002)
    hit = np.zeros(n, bool)
    alive = np.ones(n, bool)
    for _ in range(steps):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        p = o + d[idx] * t[idx, None]
        dist = sdf(p)
        h = dist < 4e-4 * (1 + t[idx])
        out = (t[idx] > tmax) | (p[:, 2] > BACK_Z + WALL_T + 0.05)
        hit[idx[h]] = True
        alive[idx[h | out]] = False
        go = ~(h | out)
        t[idx[go]] += np.maximum(dist[go] * 0.85, 1e-4)
    return t, hit


def normals(p):
    e = 4e-4
    k = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], float)
    n = sum(k[i] * sdf(p + e * k[i])[:, None] for i in range(4))
    return n / length(n)[:, None]


def soft_shadow(p, l, tmax, k=12.0, steps=48):
    """Soft visibility from p towards direction l (rows), up to tmax."""
    res = np.ones(len(p))
    t = np.full(len(p), 0.02)
    alive = np.ones(len(p), bool)
    tmax = np.broadcast_to(tmax, (len(p),))
    for _ in range(steps):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        h = sdf(p[idx] + l[idx] * t[idx, None])
        res[idx] = np.minimum(res[idx], np.clip(k * h / t[idx], 0, 1))
        t[idx] += np.clip(h, 0.01, 0.2)
        alive[idx[(res[idx] < 0.01) | (t[idx] > tmax[idx])]] = False
    return res ** 1.5


def ao(p, n):
    occ, w = 0.0, 1.0
    for i in range(1, 6):
        h = 0.03 * i
        occ = occ + w * np.clip(h - sdf(p + n * h), 0, None)
        w *= 0.6
    return np.clip(1 - 3.2 * occ, 0, 1)


# ================================================================ lights
TO_MOON = V(1.0, 0.78, 1.0)
TO_MOON /= np.linalg.norm(TO_MOON)
MOON_COL = V(0.62, 0.72, 0.95)
LAMP_COL = V(1.0, 0.6, 0.28)
SCREEN_COL = V(0.3, 0.75, 0.95)


def through_window(p, l=TO_MOON):
    """1 where a ray from p towards l leaves through a pane, 0 where wall or bar blocks it."""
    t = (WIN_Z - p[:, 2]) / l[2]
    q = p + l * t[:, None]
    inside = -hole2d(q[:, 0], q[:, 1])
    bar = bars2d(q[:, 0], q[:, 1])
    return np.clip(inside / 0.02, 0, 1) * np.clip(bar / 0.012, 0, 1) * (t > 0)


def albedo(name, p, book):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    c = np.ones((len(p), 3))
    if name == "floor":
        plank = np.floor(x / 0.19)
        seam = np.clip(np.abs((x / 0.19) % 1 - 0.5) * 2 - 0.9, 0, 1) * 6
        off = np.floor(z / 1.3 + plank * 0.37)
        grain = 0.85 + 0.15 * np.sin(z * 9 + plank * 3.1 + np.sin(z * 2.3 + plank) * 2)
        tone = 0.75 + 0.25 * np.sin(plank * 12.9898 + off * 78.233) ** 2
        return V(0.3, 0.19, 0.12)[None] * (grain * tone * (1 - 0.6 * seam))[:, None]
    if name == "wall":
        return V(0.4, 0.36, 0.34)[None] * (0.94 + 0.06 * np.sin(x * 37 + y * 23) * np.sin(z * 29 - y * 17))[:, None]
    if name == "desk":
        return V(0.45, 0.3, 0.19)[None] * (0.88 + 0.12 * np.sin(x * 31 + np.sin(z * 7) * 3))[:, None]
    if name == "book":
        return BOOK_COL[book]
    if name == "rug":
        u = np.abs(x + 0.9) / 1.3
        v = np.abs(z - 1.6) / 0.95
        ring = np.maximum(u, v)
        band = (np.sin(ring * 30) > 0.3) & (ring > 0.78)
        diamond = ((np.abs(u * 4 % 1 - 0.5) + np.abs(v * 3 % 1 - 0.5)) < 0.3) & (ring < 0.7)
        return np.where((band | diamond)[:, None], V(0.62, 0.48, 0.3)[None], V(0.45, 0.13, 0.12)[None])
    if name == "cork":
        c = V(0.55, 0.4, 0.25)[None] * (0.85 + 0.15 * np.sin(y * 113) * np.sin(z * 97))[:, None]
        for yy, zz, col in ((1.75, 1.2, V(0.9, 0.85, 0.5)), (1.45, 1.75, V(0.6, 0.8, 0.9)),
                            (1.62, 2.05, V(0.95, 0.95, 0.92)), (1.32, 1.45, V(0.9, 0.6, 0.55))):
            c[(np.abs(y - yy) < 0.11) & (np.abs(z - zz) < 0.13)] = col
        return c
    flat = {"frame": V(0.5, 0.48, 0.44), "trim": V(0.5, 0.48, 0.44), "sill": V(0.55, 0.52, 0.47),
            "bezel": V(0.07, 0.07, 0.07), "screen": V(0.02, 0.02, 0.02), "keys": V(0.18, 0.18, 0.18),
            "lamp": V(0.75, 0.32, 0.18), "bulb": V(1, 1, 1), "mug": V(0.75, 0.73, 0.68),
            "board": V(0.08, 0.18, 0.42), "shelf": V(0.38, 0.26, 0.17), "pot": V(0.6, 0.33, 0.22),
            "leaf": V(0.2, 0.45, 0.2), "chair": V(0.12, 0.12, 0.12), "hoodie": V(0.62, 0.36, 0.24),
            "skin": V(0.75, 0.52, 0.4), "hair": V(0.45, 0.29, 0.16), "phones": V(0.1, 0.1, 0.1),
            "cat": V(0.05, 0.05, 0.05)}
    return c * flat[name]


def shade(p, n, mat, book):
    """Radiance leaving each hit point back towards the camera."""
    col = np.zeros((len(p), 3))
    alb = np.zeros((len(p), 3))
    for i, name in enumerate(NAMES):
        m = mat == i
        if m.any():
            alb[m] = albedo(name, p[m], book[m])
    occl = ao(p, n)
    q = p + n * 2e-3
    # moon through the window
    vis = through_window(q)
    lit = vis > 0
    if lit.any():
        tw = (WIN_Z - q[lit, 2]) / TO_MOON[2]
        vis[lit] *= soft_shadow(q[lit], np.repeat(TO_MOON[None], lit.sum(), 0), tw, k=24)
    col += alb * MOON_COL * (2.4 * vis * np.clip(n @ TO_MOON, 0, 1))[:, None]
    # lamp: a cone pointing down
    lv = LAMP_BULB - q
    ld = length(lv)
    l = lv / ld[:, None]
    cone = np.clip((-l[:, 1] - 0.25) / 0.35, 0, 1)
    sh = soft_shadow(q, l, ld - 0.06, k=10)
    col += alb * LAMP_COL * (2.2 * cone * sh * np.clip((n * l).sum(1), 0, 1) / (ld ** 2 + 0.02))[:, None]
    # monitor glow, only in front of the screen
    sv = MON_C + V(0, 0, -0.08) - q
    sd = length(sv)
    s = sv / sd[:, None]
    front = np.clip(s[:, 2], 0, 1) ** 0.5
    sh2 = soft_shadow(q, s, sd - 0.1, k=6)
    col += alb * SCREEN_COL * (0.3 * front * sh2 * np.clip((n * s).sum(1), 0, 1) / (sd ** 2 + 0.05))[:, None]
    # fairy lights: small, warm, unshadowed
    for i, fp in enumerate(FAIRY):
        fv = fp + V(0, -0.02, -0.03) - q
        fd = length(fv)
        col += alb * FAIRY_COL[i % 4] * (0.0025 * np.clip((n * fv).sum(1) / fd, 0, 1) / (fd ** 2 + 0.01))[:, None]
    # ambient night-blue from the window
    sky = np.clip(0.5 + 0.5 * n[:, 2], 0, 1) * 0.05 + 0.014
    col += alb * (V(0.4, 0.46, 0.7) * sky[:, None]) * occl[:, None]
    name = np.array(NAMES)[mat]
    # moonlight bouncing off the floor picks out the figure: a soft fill and a rim
    who = np.isin(name, ["hoodie", "hair", "skin", "phones"])
    if who.any():
        v = CAM - p[who]
        v /= length(v)[:, None]
        nw = n[who]
        fill = np.clip(nw @ (V(0.8, 0.35, -0.5) / np.linalg.norm(V(0.8, 0.35, -0.5))), 0, 1)
        rim = np.clip(1 - np.abs((nw * v).sum(1)), 0, 1) ** 2.5
        col[who] += alb[who] * V(0.6, 0.66, 0.85) * (0.35 * fill)[:, None] + V(0.5, 0.58, 0.8) * (0.25 * rim)[:, None]
    col[name == "bulb"] = V(1.6, 1.15, 0.7) * 3
    col[name == "screen"] = V(0.05, 0.14, 0.17)
    return col


def beam(o, d, t_end, n=48):
    """Moonlight scattered by the air along each camera ray, plus lamp and screen halos."""
    acc = np.zeros(len(d))
    dt = t_end / n
    for i in range(n):
        s = (i + 0.5) * dt
        p = o + d * s[:, None]
        acc += through_window(p) * (p[:, 2] < BACK_Z) * dt
    light = MOON_COL[None] * (0.005 * acc)[:, None]
    for c, colr, k, r in ((LAMP_BULB, V(1.0, 0.65, 0.3), 0.06, 0.1),
                          (MON_C + V(0, 0, -0.05), SCREEN_COL, 0.012, 0.45)):
        s = np.clip(((c - o) * d).sum(1), 0, t_end)
        dist = length(o + d * s[:, None] - c)
        light += colr[None] * (k / (1 + (dist / r) ** 2))[:, None]
    return light


# ================================================================ sky through the window
def hills(u, base, amp, seed):
    a = np.zeros_like(u)
    for k, f in enumerate((7, 13, 29, 61)):
        a += np.sin(u * f + seed * (k + 1) * 1.7) / (k + 1)
    return base + amp * a


TURBINES = [(0.0, 1.6), (0.13, 1.9), (0.28, 1.45), (0.4, 1.2), (0.49, 1.0)]   # (u, size)


def ridge(u):
    return hills(u, 0.0, 0.012, 2.0)


def seg_dist(u, v, a, b):
    pa = np.stack([u - a[0], v - a[1]], 1)
    ba = np.asarray(b) - np.asarray(a)
    h = np.clip(pa @ ba / (ba @ ba), 0, 1)
    return np.hypot(*(pa - h[:, None] * ba).T), h


def sky_colour(dirs, frame, slow=None):
    """Colour seen through the window in each direction, at a given frame.

    Stars twinkle on the slow clock `slow` (a frame number), everything else
    moves on `frame`."""
    slow = frame if slow is None else slow
    u = np.arctan2(dirs[:, 0], dirs[:, 2])
    v = np.arcsin(np.clip(dirs[:, 1], -1, 1))
    t = frame / FPS
    loop = FRAMES / FPS
    # gradient: a faint glow at the horizon, deep blue overhead
    hor = np.exp(-np.clip(v, 0, None) / 0.07)
    col = V(0.006, 0.012, 0.045)[None] + V(0.1, 0.12, 0.22)[None] * hor[:, None]
    col += V(0.06, 0.04, 0.05)[None] * np.exp(-np.clip(v, 0, None) / 0.025)[:, None]
    # milky way
    band = np.exp(-((v - (0.36 - 0.5 * u)) / 0.05) ** 2)
    col += V(0.02, 0.024, 0.04)[None] * (band * (0.6 + 0.4 * np.sin(u * 90) * np.sin(v * 70)))[:, None]
    # stars, on a jittered grid in angle space; each twinkles at a whole number of cycles per loop
    g = 0.0125
    iu, iv = np.floor(u / g), np.floor(v / g)
    h1 = np.sin(iu * 127.1 + iv * 311.7) * 43758.5453
    h1 -= np.floor(h1)
    h2 = np.sin(iu * 269.5 + iv * 183.3) * 43758.5453
    h2 -= np.floor(h2)
    su, sv = (iu + 0.2 + 0.6 * h1) * g, (iv + 0.2 + 0.6 * h2) * g
    dist = np.hypot(u - su, v - sv)
    dens = 0.93 - 0.06 * band
    mag = np.clip((h1 - dens) / (1 - dens), 0, 1)
    cycles = 1 + np.floor(h2 * 3)
    twinkle = 0.6 + 0.4 * np.sin(2 * math.pi * cycles * slow / FRAMES + h1 * 50)
    star = (h1 > dens) * (sv > 0.05) * np.exp(-(dist / 0.0024) ** 2) * (2.0 + 7.0 * mag) * twinkle
    col += V(0.85, 0.9, 1.0)[None] * star[:, None]
    # the moon, a crescent with a halo
    mu, mv, mr = 0.37, 0.3, 0.05
    dm = np.hypot(u - mu, v - mv)
    lit = np.clip((mr - dm) / 0.003, 0, 1) * np.clip((np.hypot(u - mu + 0.024, v - mv - 0.015) - mr * 0.9) / 0.003, 0, 1)
    col += V(1.0, 0.92, 0.75)[None] * (3.0 * lit)[:, None]
    col += V(0.15, 0.17, 0.25)[None] * (0.6 * np.exp(-dm / 0.05))[:, None]
    # clouds, lit from above by the moon
    cl = np.zeros_like(u)
    for k, (f, a) in enumerate(((11, 1.0), (23, 0.5), (47, 0.25), (93, 0.12))):
        cl += a * np.sin(u * f + np.sin(v * f * 1.7 + k) + k * 3.1) * np.sin(v * f * 2.1 + u * f * 0.3 + k)
    layer = np.exp(-((v - 0.15) / 0.035) ** 2) + 0.7 * np.exp(-((v - 0.24) / 0.02) ** 2)
    dc = np.clip((cl * 0.6 + 0.25) * layer, 0, 1)
    edge = np.clip(dc * 3, 0, 1) * (1 - np.clip(dc * 2 - 0.4, 0, 1))
    cloud = V(0.05, 0.065, 0.13)[None] * (1 - dc[:, None]) + V(0.15, 0.16, 0.24)[None] * dc[:, None]
    col = col * (1 - dc[:, None]) + cloud * dc[:, None]
    col += V(0.5, 0.46, 0.4)[None] * (0.45 * edge * np.exp(-np.abs(u - mu) / 0.25))[:, None]
    # shooting stars, twice per loop
    for s0, s1, a0, dirn in ((18, 27, (0.06, 0.37), (0.17, -0.08)), (46, 53, (0.42, 0.33), (-0.14, -0.07))):
        if not s0 <= frame < s1:
            continue
        a0, dirn = np.array(a0), np.array(dirn)
        k = (frame - s0 + 1) / (s1 - s0)
        head = a0 + dirn * k
        dd, hh = seg_dist(u, v, head - dirn * 0.4, head)
        fade = math.sin(k * math.pi * 0.95)
        col += V(0.9, 0.95, 1.0)[None] * (6.0 * fade * hh ** 2 * np.exp(-(dd / 0.003) ** 2))[:, None]
    # far ridge with turbines; they turn a whole number of thirds per loop
    far = v < ridge(u)
    for i, (tu, sz) in enumerate(TURBINES):
        hub = np.array([tu, ridge(np.array([tu]))[0] + 0.075 * sz])
        far |= (np.abs(u - tu) < 0.002 * sz + 0.0008) & (v > hub[1] - 0.08 * sz) & (v < hub[1])
        rot = -2 * math.pi / 3 * (4 + i % 2) * t / loop + i * 1.3
        for b in range(3):
            a = rot + b * 2 * math.pi / 3
            dd, hh = seg_dist(u, v, hub, hub + 0.05 * sz * np.array([math.cos(a), math.sin(a)]))
            far |= dd < 0.0034 * sz * (1.3 - hh)
        if (frame // 4 + i) % 4 == 0:
            dl = np.hypot(u - hub[0], v - hub[1] - 0.004)
            col += V(1.0, 0.08, 0.05)[None] * (1.6 * np.exp(-(dl / 0.004) ** 2))[:, None]
    col[far] = col[far] * 0.12 + V(0.03, 0.035, 0.06)
    nh = hills(u, -0.045, 0.02, 5.0)
    near = v < nh
    col[near] = V(0.012, 0.014, 0.022)
    # lit windows on the near hill
    wu = np.floor(u / 0.009)
    wh = np.sin(wu * 91.7) * 9431.7
    wh -= np.floor(wh)
    win = near & (wh > 0.86) & (np.abs(v - (nh - 0.012 - 0.01 * wh)) < 0.0028) & (np.abs(u - (wu + 0.5) * 0.009) < 0.002)
    col[win] = V(1.0, 0.7, 0.35) * 1.2
    # two cars on the road along the near hill, one each way, once per loop
    for ph, sgn, c in ((t / loop, 1, V(1.0, 0.9, 0.7)), ((t / loop + 0.5) % 1, -1, V(1.0, 0.15, 0.08))):
        cu = -0.17 + 0.75 * (ph if sgn > 0 else 1 - ph)
        cv = hills(np.array([cu]), -0.045, 0.02, 5.0)[0] - 0.006
        for off in (-0.0025, 0.0025):
            dl = np.hypot(u - cu - off, v - cv)
            col += c[None] * (2.5 * np.exp(-(dl / 0.0022) ** 2))[:, None]
    return col


# ================================================================ render
def tonemap(c):
    c = np.maximum(c, 0) * 4.0
    c = c / (1 + c)
    return c ** (1 / 1.6)


def render_rays(d):
    """Colour along each ray (without the sky, which is added per frame) and whether it reached the sky."""
    t, hit = march(CAM, d)
    col = np.zeros((len(d), 3))
    h = np.nonzero(hit)[0]
    p = CAM + d[h] * t[h, None]
    mat, book = material(p)
    col[h] = shade(p, normals(p), mat, book)
    t_air = np.where(hit, t, (BACK_Z - CAM[2]) / np.maximum(d[:, 2], 1e-3))
    col += beam(CAM, d, np.minimum(t_air, 12))
    return col, ~hit


# the moving bits: the head nods to the beat (4 frames), the hands type (8 frames),
# the cat's tail sways on the slow clock
NOD = [0.0, 0.5, 1.0, 0.5]
TYPE_L = [1, 0, 0, 1, 0, 0, 1, 0]
TYPE_R = [0, 0, 1, 0, 1, 0, 0, 1]
TAILS = 16


def person_pose(i):
    return {"nod": NOD[i % 4], "tl": TYPE_L[i], "tr": TYPE_R[i], "tail": 0.0}


def cat_pose(i):
    return {"nod": 0.0, "tl": 0.0, "tr": 0.0, "tail": 0.55 * math.sin(2 * math.pi * i / TAILS)}


def render():
    d = camera_rays()
    print("marching", len(d), "rays")
    t, hit = march(CAM, d)
    p = CAM + d * t[:, None]
    col = np.zeros((len(d), 3))
    idx = np.nonzero(hit)[0]
    mat, book = material(p[idx])
    n = normals(p[idx])
    print("shading")
    col[idx] = shade(p[idx], n, mat, book)
    t_air = np.where(hit, t, (BACK_Z - CAM[2]) / np.maximum(d[:, 2], 1e-3))
    col += beam(CAM, d, np.minimum(t_air, 12))
    mats = np.full(len(d), -1)
    mats[idx] = mat
    return dict(col=col, sky=~hit, dirs=d, mats=mats)


def by_cell(a):
    return a.reshape(ROWS, SY, COLS, SX, *a.shape[2:])


DENSITY = " .'`,:;-~=+i!*co%x#&@"
_cov = Matcher(DENSITY).masks.mean(1)
DENS_CH = np.array(list(DENSITY))[np.argsort(_cov)]
DENS_COV = np.sort(_cov)                 # share of the cell each glyph inks, 0 .. 0.34


def blur(a, r=2):
    p = np.pad(a, r, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(2 * r + 1) for j in range(2 * r + 1)) / (2 * r + 1) ** 2


LUMA = np.array([0.3, 0.55, 0.15])


def to_cells(rgb, scale=0.75, expo=0.92, blank=0.03, sat=1.05, sharp=0.6, contrast=1.5):
    """Tone-mapped picture (H, W, 3) -> glyphs and the colour each glyph is printed in.

    Seen from a distance a cell looks like its glyph's ink coverage times the
    ink colour. So for every cell the glyph is chosen by how bright the cell
    should look (sparse glyphs for dim cells, dense ones for bright cells) and
    the ink is the cell's colour divided by that glyph's coverage: the printed
    page then averages back to the rendered picture, with the glyphs as grain."""
    T = by_cell(rgb).mean((1, 3))
    T = T * ((T @ LUMA) ** (contrast - 1))[..., None]      # deepen the shadows, keep the lights
    lt = T @ LUMA
    ls = np.clip(lt + sharp * (lt - blur(lt)), 0, None)          # a little local contrast
    P = T * (ls / np.maximum(lt, 1e-4))[..., None] * scale
    g = P @ LUMA
    P = np.clip(g[..., None] + sat * (P - g[..., None]), 0, None)
    mx = P.max(-1)
    top = DENS_COV[-1]
    want = np.maximum(top * np.clip(mx / top, 0, 1) ** expo, mx)
    k = np.abs(DENS_COV[None, None] - want[..., None]).argmin(-1)
    ch = DENS_CH[k]
    ink = np.clip(P / np.maximum(DENS_COV[k], 1e-3)[..., None], 0, 1)
    # neighbours share a hue: blend each ink towards its 3x3 average, weighted by brightness
    wgt = ink.max(-1) + 1e-3
    avg = np.stack([blur(ink[..., i] * wgt, 1) for i in range(3)], -1) / blur(wgt, 1)[..., None]
    ink = np.clip(0.45 * ink + 0.55 * avg * (ink.max(-1) / np.maximum(avg.max(-1), 1e-3))[..., None], 0, 1)
    ch[mx < blank] = " "
    return ch, ink, P


def quantise(colours, k=44, iters=24):
    pts = colours.reshape(-1, 3)
    cent = pts[rng.choice(len(pts), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((pts[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for i in range(k):
            m = lab == i
            cent[i] = pts[m].mean(0) if m.any() else pts[rng.integers(len(pts))]
    return cent


# ---------------------------------------------------------------- overlays
def monitor(ch, cl, mask, frame):
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()

    def put(r, c, s, k):
        vis = np.nonzero(mask[r])[0]
        c = max(c, vis.min() + 1) if len(vis) else c
        for i, x in enumerate(s):
            if c + i <= c1 and mask[r, c + i]:
                ch[r, c + i], cl[r, c + i] = x, k

    ch[mask], cl[mask] = ":", "sb"
    put(r0, c0 + 1, "arda@jetson:~$ ./monitor turbine_03", "sd")
    for i in range(c1 - c0 - 1):          # a vibration trace scrolling right to left
        x = (i + frame * 2) * 0.55
        y = 0.5 * math.sin(x) + 0.3 * math.sin(2.7 * x + 1) + 0.15 * math.sin(7.1 * x)
        lvl = int(np.clip((y + 0.95) / 1.9 * 7, 0, 6.99))
        put(r0 + 2 if lvl < 4 else r0 + 1, c0 + 1 + i, "_.-~'^`"[lvl], "sc")
    put(r1 - 1, c0 + 1, f"rms {0.40 + 0.02 * math.sin(2 * math.pi * frame / FRAMES):.2f} g", "sd")
    put(r1, c0 + 1, "nominal", "sc")


NEON_AT = V(-1.75, 2.62, BACK_Z)          # centre of the sign on the back wall
NEON_COL = V(1.0, 0.3, 0.5)
# frame -> brightness; the sign hums, stutters once, and its Ç is a little loose
NEON_FLICKER = {21: 0.1, 22: 0.7, 23: 0.05, 24: 0.05, 25: 0.8, 53: 0.45}
C_FLICKER = {36: 0.0, 37: 1.0, 38: 0.0, 39: 0.0, 40: 0.6}


def neon_cells():
    """[(row, col, char, letter index)] of a small neon line with the name, centred on NEON_AT."""
    x, y = project(NEON_AT)
    word = "arda çabuk"
    c0, r0 = int(round(x / SX - len(word) / 2)), int(round(y / SY))
    return [(r0, c0 + i, ch, i) for i, ch in enumerate(word) if ch != " "]


def neon_level(frame, letter):
    b = NEON_FLICKER.get(frame, 1.0)
    if letter == 5:
        b *= C_FLICKER.get(frame, 1.0)
    return b


def neon_glow(cells):
    """Pink light the sign throws on the wall around it (kept steady: only the tubes flicker)."""
    m = np.zeros((ROWS, COLS))
    for r, c, _, _ in cells:
        m[r, c] = 1.0
    m = np.kron(m, np.ones((SY, SX)))
    g = blur(blur(m, 6), 6)
    return (g[..., None] * NEON_COL * 0.45).reshape(-1, 3)


def project(P):
    """3-D point -> (x, y) on the ray grid."""
    f = LOOK - CAM
    f /= np.linalg.norm(f)
    r = np.cross(V(0, 1, 0), f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    v = P - CAM
    z = v @ f
    tx = math.tan(HFOV / 2)
    x = (v @ r / z / tx + 1) / 2 * W - 0.5
    y = (-(v @ u) / z / (tx * H / W) + 1) / 2 * H - 0.5
    return x, y


YY, XX = np.mgrid[:H, :W]
_mr = np.random.default_rng(11)
MOTES = []
while len(MOTES) < 6:
    _b = V(_mr.uniform(-2.2, 1.8), _mr.uniform(0.4, 2.4), _mr.uniform(0.8, 3.2))
    if through_window(_b[None])[0] > 0.5:
        MOTES.append((_b, _mr.random(), int(_mr.integers(1, 3))))
LED = BOARD + V(0.08, 0.012, -0.04)


def glows(frame, slow=None):
    """Light sources that change per frame, added to the picture as soft sprites."""
    out = np.zeros((H, W, 3))
    slow = frame if slow is None else slow
    t = slow / FRAMES
    lights = []
    for i, fp in enumerate(FAIRY):
        k = 1 + i % 3
        lights.append((fp, FAIRY_COL[i % 4] * (0.75 + 0.45 * math.sin(2 * math.pi * k * t + i * 2.1)), 1.0))
    if (frame // 4) % 2 == 0:
        lights.append((LED, V(0.3, 1.0, 0.4) * 1.4, 0.8))
    ft = frame / FRAMES
    for b, ph, k in MOTES:                 # dust drifting through the moonbeam
        P = b + 0.18 * V(math.sin(2 * math.pi * (ft + ph)), -abs(math.sin(math.pi * (k * ft + ph))) * 0.8,
                         math.cos(2 * math.pi * (ft + ph)) * 0.6)
        lit = through_window(P[None])[0]
        if lit > 0:
            lights.append((P, MOON_COL * 0.9 * lit, -0.6))
    for P, c, s in lights:
        halo = s > 0                       # a negative size means a bare point, no halo
        s = abs(s)
        x, y = project(P)
        x0, x1, y0, y1 = int(x) - 14, int(x) + 15, int(y) - 14, int(y) + 15
        d2 = (XX[y0:y1, x0:x1] - x) ** 2 + (YY[y0:y1, x0:x1] - y) ** 2
        g = 3.0 * np.exp(-d2 / (2 * (1.0 * s) ** 2)) + halo * 0.05 * np.exp(-d2 / (2 * (5.0 * s) ** 2))
        out[y0:y1, x0:x1] += g[..., None] * c
    return out.reshape(-1, 3)


def build():
    cache = pathlib.Path(__file__).with_name("_preview") / "hero_render.npz"
    if cache.exists() and "--rerender" not in sys.argv:
        z = np.load(cache)
        R = {k: z[k] for k in z.files}
    else:
        R = render()
        cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache, **R)
    col, sky, dirs, mats = R["col"], R["sky"], R["dirs"], R["mats"]

    # re-render the rays around the figure and the cat for every pose
    img = mats.reshape(H, W)
    who = np.isin(img, [NAMES.index(n) for n in ("hair", "phones", "skin", "hoodie")])
    pidx = np.nonzero((blur(who.astype(float), 8) > 0).ravel())[0]
    cidx = np.nonzero((blur((img == NAMES.index("cat")).astype(float), 14) > 0).ravel())[0]
    pcache = cache.with_name("hero_poses.npz")
    if pcache.exists() and "--rerender" not in sys.argv and "--reposes" not in sys.argv:
        z = np.load(pcache)
        PC, PS, CC, CS = z["pc"], z["ps"], z["cc"], z["cs"]
    else:
        PC, PS, CC, CS = [], [], [], []
        for i in range(8):
            POSE.update(person_pose(i))
            c_, s_ = render_rays(dirs[pidx])
            PC.append(c_), PS.append(s_)
        for i in range(TAILS):
            POSE.update(cat_pose(i))
            c_, s_ = render_rays(dirs[cidx])
            CC.append(c_), CS.append(s_)
        POSE.update(person_pose(0))
        PC, PS, CC, CS = map(np.array, (PC, PS, CC, CS))
        np.savez_compressed(pcache, pc=PC, ps=PS, cc=CC, cs=CS)
    print(f"poses: {len(pidx)} + {len(cidx)} rays")

    sign = neon_cells()
    halo = neon_glow(sign)
    pics = []
    for f in range(FRAMES):
        slow = f - f % SLOW                    # twinkles move on a slower clock
        c, sk = col.copy(), sky.copy()
        c[pidx], sk[pidx] = PC[f % 8], PS[f % 8]
        c[cidx], sk[cidx] = CC[(f // SLOW) % TAILS], CS[(f // SLOW) % TAILS]
        c[sk] += sky_colour(dirs[sk], f, slow)
        c += glows(f, slow) + halo
        ch, ink, P = to_cells(tonemap(c).reshape(H, W, 3))
        if f:                                   # keep cells whose target barely moved as they were
            same = np.abs(P - pics[0][2]).max(-1) < 0.02
            ch[same], ink[same] = pics[0][0][same], pics[0][1][same]
        pics.append((ch, ink, P))

    cent = quantise(np.concatenate([ink[ch != " "] for ch, ink, _ in pics[::8]]))
    names = [f"{'cdefghijlmnopq'[i // 10]}{i % 10}" for i in range(len(cent))]
    palette = {n: "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c) for n, c in zip(names, cent)}
    palette.update({"sc": "#e4fff9", "sd": "#a6efe2", "sb": "#2f9a8f", "ld": "#5dff7a", "lo": "#1d4a26", "st": "#7d8794",
                    "na": "#ff9cc2", "nm": "#b8467a", "nb": "#4a1f33"})
    names = np.array(names)

    m = by_cell(mats.reshape(H, W))
    screen = (m == NAMES.index("screen")).mean((1, 3)) > 0.6
    mug = np.argwhere((m == NAMES.index("mug")).mean((1, 3)) > 0.3)
    mr, mc = mug[:, 0].min(), int(round(mug[:, 1].mean()))

    grids = []
    for f, (ch, ink, _) in enumerate(pics):
        cl = names[((ink[..., None, :] - cent[None, None]) ** 2).sum(-1).argmin(-1)].astype("<U2")
        ch = ch.copy()
        monitor(ch, cl, screen, f)
        rs, cs = [r for r, _, _, _ in sign], [c for _, c, _, _ in sign]
        ch[min(rs):max(rs) + 1, min(cs) - 2:max(cs) + 3] = " "      # a dark strip behind the tube
        for r, c, g, i in sign:
            b = neon_level(f, i)
            ch[r, c], cl[r, c] = g, ("na" if b > 0.75 else "nm" if b > 0.3 else "nb")
        for k in range(3):                 # steam: three wisps rising and swaying
            ph = (f / FRAMES * 4 + k / 3) % 1
            if ph < 0.85:
                r = mr - 1 - int(ph * 3)
                c = mc + int(round(math.sin(ph * 6 + k) * 1.2))
                ch[r, c], cl[r, c] = "'()~"[int(ph * 4)], "st"
        cl[ch == " "] = ""
        grids.append((ch, cl))

    frames = []
    for ch, cl in grids:
        g = Grid(COLS, ROWS)
        g.paste(ch, cl)
        frames.append(g)
    static, sprites = pack(frames)
    print(f"{len(sprites)} sprites")

    movie = Movie(COLS, ROWS, pad=(12, 14), palette=palette,
                  title="Arda Çabuk — embedded software and machine learning",
                  desc="A ray-marched room at night, drawn in ASCII. A small pink neon on the wall reads arda çabuk "
                       "and stutters now and then. Under it someone with light-brown hair, a rust hoodie and headphones sits at a "
                       "desk with their back to you, nodding to the music and typing; the monitor reads "
                       "arda@jetson and shows a scrolling vibration trace marked nominal, a "
                       "development board blinks beside the keyboard, steam rises from a mug under a desk lamp. "
                       "Through a tall arched window a cat sits on the sill against a crescent moon, stars "
                       "twinkle, shooting stars cross, cars pass on the hill road and wind turbines turn on the ridge "
                       "with red lights blinking. Fairy lights twinkle under the ceiling and dust drifts in the "
                       "moonlight falling through the window.")
    movie.add(static)
    for fr, hold in sprites:
        movie.add(fr, FPS, hold=hold)
    return movie, pics


if __name__ == "__main__":
    m, _ = build()
    if "--preview" in sys.argv:
        m.preview("hero.png", times=(0, 2.5, 5.0))
    m.write("hero.svg")
