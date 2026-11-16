"""Hero: a room at three in the morning, ray-marched and printed in ASCII.

The room is a real 3-D scene made of signed distance functions: walls with an
arched window cut through the back one, a desk, a monitor, a lamp, a shelf of
books, a plant, a cat on the sill and someone in profile at the keyboard. It
is lit by three lights: the moon through the window (with its beam scattered
in the air), an amber desk lamp aimed at the face and hands, and the pale glow
of the monitor.

The rays give an 800 x 448 colour picture, which is printed as 240 x 67
characters. A night scene only works in print if the dark stays empty, so the
ambient gloom is crushed to nothing; what is lit gets a bright ink in its own
hue and a glyph that is denser the lighter the cell is.

The camera never moves, so the room is rendered once. What moves is redrawn
per frame on top: the sky through the window (stars, a shooting star, wind
turbines turning on the ridge with their red lights), the vibration trace on
the monitor, the board's blinking LED and the steam from the mug.
"""
import math
import pathlib
import sys

import numpy as np

from PIL import Image

from engine import Grid, Matcher, Movie, pack

W, H = 800, 448                        # rays: the picture the room is rendered at
COLS, ROWS = 240, 67                   # character cells it is printed in
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


def cyl_z(p, c, r, hh):
    """A cylinder lying along z (the bass drum)."""
    d = np.stack([np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - r, np.abs(p[:, 2] - c[2]) - hh], -1)
    return np.minimum(d.max(-1), 0) + length(np.maximum(d, 0))


def disc(p, c, r, n, t=0.006):
    """A thin disc (a cymbal) centred at c with normal n."""
    q = p - c
    a = q @ n
    rad = length(q - a[:, None] * n[None])
    d = np.stack([rad - r, np.abs(a) - t], -1)
    return np.minimum(d.max(-1), 0) + length(np.maximum(d, 0))


def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = length(q)
    k1 = length(q / r)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-6)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


# ================================================================ camera (fixed)
CAM = V(-2.9, 1.15, -0.6)
LOOK = V(-0.75, 1.5, 3.5)
HFOV = math.radians(80)


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


# the workstation stands against the left wall (x = -3.35); the person sits in
# profile facing it (towards -x); ZF is the depth everything is centred on
DESK_Y, ZF = 0.76, 1.6
MON_A = math.radians(35)                               # monitor turned this far towards the camera
MON_N = V(math.cos(MON_A), 0, -math.sin(MON_A))        # screen normal
MON_U = V(math.sin(MON_A), 0, math.cos(MON_A))         # along the screen's width
MON_C, MON_H = V(-3.1, 1.16, ZF + 0.12), V(0.36, 0.24, 0.025)
LAMP_HEAD = V(-2.88, 1.5, ZF + 0.22)
LAMP_BULB = LAMP_HEAD + V(0, -0.12, -0.02)
LAMP_AIM = V(0.75, -0.45, -0.35) / np.linalg.norm(V(0.75, -0.45, -0.35))
MUG = V(-2.78, DESK_Y + 0.06, ZF + 0.42)
BOARD = V(-3.0, DESK_Y + 0.012, ZF - 0.42)

# fairy lights draped across the wall above the desk: two sagging spans
FAIRY = []
for _a, _b in ((-3.3, -1.9), (-1.9, -0.45)):
    for _s in np.linspace(0, 1, 9)[(0 if _a < -2 else 1):]:
        FAIRY.append(V(_a + (_b - _a) * _s, 3.17 - 0.15 * math.sin(math.pi * _s), BACK_Z - 0.03))
FAIRY_COL = [V(1.0, 0.72, 0.35), V(1.0, 0.45, 0.3), V(1.0, 0.85, 0.55), V(0.9, 0.3, 0.22)]

# books: (x0, width, height, colour)
BOOKS = []
_x = -2.95
while _x < -1.15:
    _w = rng.uniform(0.045, 0.085)
    _h = rng.uniform(0.19, 0.29)
    _c = rng.choice([[0.55, 0.16, 0.12], [0.38, 0.42, 0.22], [0.62, 0.48, 0.24], [0.35, 0.35, 0.33],
                     [0.5, 0.45, 0.4], [0.4, 0.12, 0.1], [0.72, 0.6, 0.42]])
    BOOKS.append((_x, _w, _h, np.array(_c) * rng.uniform(0.75, 1.1)))
    _x += _w + rng.uniform(0.003, 0.012)
    if rng.random() < 0.12:
        _x += 0.12
BOOK_COL = np.array([b[3] for b in BOOKS])


CAT_X, CAT_S = 1.5, 1.4                 # the cat sits in the middle pane, in front of the bright horizon
CURLS = [(x_, z_) for x_ in (-0.075, -0.035, 0.005, 0.045) for z_ in (-0.05, 0.0, 0.05)]
# the drum kit stands in the moonlight on the floor, modelled at unit size around DRUMS and scaled
DRUMS, DRUM_S = V(0.1, 0, 1.6), 1.3
KIT_TOMS = [(V(-0.02, 0.66, 0.1), 0.13, 0.08), (V(-0.42, 0.6, 0.2), 0.16, 0.065),   # rack tom, snare,
            (V(0.45, 0.42, 0.25), 0.19, 0.17)]                                       # floor tom (y-cylinders)
WIN_AIM = V(1.3, 1.9, WIN_Z)


def _aim(o):
    """A cymbal tilted part of the way towards mirroring the window to the camera."""
    c = DRUMS + DRUM_S * o
    a, b = WIN_AIM - c, CAM - c
    n = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    n = 0.5 * n / np.linalg.norm(n) + 0.5 * V(0, 1, 0)
    return o, n / np.linalg.norm(n)


KIT_CYMBALS = [(V(-0.75, 0.86, 0.25), 0.17, V(0, 1, 0)), (V(-0.75, 0.9, 0.25), 0.17, V(0, 1, 0)),   # hi-hat
               (*_aim(V(0.5, 1.28, -0.05)), 0.24), (*_aim(V(0.88, 1.08, 0.3)), 0.26)]          # crash, ride
KIT_CYMBALS = [c if np.ndim(c[1]) == 0 else (c[0], c[2], c[1]) for c in KIT_CYMBALS]
POSE = {"nod": 0.0, "tl": 0.0, "tr": 0.0, "tail": 0.0}   # the moving bits, set per frame


def mon_local(p):
    """Points in the monitor's frame: (along the width, up, along the screen normal)."""
    q = p - MON_C
    return np.stack([q @ MON_U, q[:, 1], q @ MON_N], -1)


def head_frame():
    """Centre of the head and its tilt (radians, forward and down) for the current nod."""
    nod = POSE["nod"]
    return V(-2.32 - 0.04 * nod, 1.33 - 0.05 * nod, ZF), 0.28 * nod


def head_local(p, c, tilt):
    """Points relative to the head, un-tilted: -x is forward, y is up."""
    q = p - c
    cs, sn = math.cos(tilt), math.sin(tilt)
    return np.stack([cs * q[:, 0] + sn * q[:, 1], -sn * q[:, 0] + cs * q[:, 1], q[:, 2]], -1)


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
    # desk against the left wall: top, two side panels, a drawer block at the far end
    d = box(p, V(-2.95, DESK_Y - 0.02, ZF + 0.18), V(0.4, 0.022, 0.78))
    for lz in (ZF - 0.57, ZF + 0.93):
        d = np.minimum(d, box(p, V(-2.98, DESK_Y / 2, lz), V(0.34, DESK_Y / 2, 0.03)))
    d = np.minimum(d, box(p, V(-3.0, 0.5, ZF + 0.66), V(0.3, 0.22, 0.22)))
    out.append(("desk", d))
    # monitor, turned towards the room
    q = mon_local(p)
    out.append(("bezel", np.minimum.reduce([
        rbox(q, V(0, 0, 0), MON_H, 0.015),
        box(q, V(0, -0.3, -0.03), V(0.03, 0.1, 0.02)),
        box(q, V(0, DESK_Y + 0.008 - MON_C[1], -0.04), V(0.15, 0.008, 0.1))])))
    out.append(("screen", box(q, V(0, 0, MON_H[2]), V(MON_H[0] - 0.03, MON_H[1] - 0.03, 0.006))))
    out.append(("keys", rbox(p, V(-2.8, DESK_Y + 0.012, ZF - 0.02), V(0.08, 0.012, 0.24), 0.008)))
    # lamp at the far end of the desk, its head leaning over the keyboard
    base = V(-3.15, DESK_Y, ZF + 0.72)
    elbow = V(-3.2, 1.66, ZF + 0.55)
    head = LAMP_HEAD
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
    out.append(("board", box(p, BOARD, V(0.07, 0.008, 0.12))))
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
    # chair, back rest on the +x side
    out.append(("chair", np.minimum.reduce([
        rbox(p, V(-2.1, 0.48, ZF), V(0.23, 0.035, 0.23), 0.02),
        rbox(p, V(-1.84, 0.8, ZF), V(0.03, 0.24, 0.21), 0.03),
        capsule(p, V(-1.88, 0.5, ZF), V(-1.86, 0.6, ZF), 0.02),
        capsule(p, V(-2.1, 0.05, ZF), V(-2.1, 0.46, ZF), 0.025),
        capsule(p, V(-2.4, 0.03, ZF), V(-1.8, 0.03, ZF), 0.02),
        capsule(p, V(-2.1, 0.03, ZF - 0.3), V(-2.1, 0.03, ZF + 0.3), 0.02)])))
    # the person, in profile, facing the desk (-x)
    hip, chest = V(-2.06, 0.64, ZF), V(-2.2, 1.06, ZF)
    torso = smin(capsule(p, hip + V(0, 0, -0.07), chest + V(0, 0, -0.08), 0.13),
                 capsule(p, hip + V(0, 0, 0.07), chest + V(0, 0, 0.08), 0.13), 0.05)
    torso = smin(torso, capsule(p, chest + V(0, 0, -0.16), chest + V(0, 0, 0.16), 0.085), 0.06)
    torso = smin(torso, ellipsoid(p, chest + V(0.09, 0.07, 0), V(0.08, 0.06, 0.12)), 0.03)    # the hood
    arms, wrists = [], []
    for side, k in ((-1, POSE["tl"]), (1, POSE["tr"])):
        sh_ = chest + V(-0.02, -0.02, 0.19 * side)
        el = V(-2.3, 0.8 + 0.015 * k, ZF + 0.21 * side)
        wr = V(-2.66 - 0.02 * k, 0.815 + 0.06 * k, ZF + 0.1 * side)
        arms += [capsule(p, sh_, el, 0.06), capsule(p, el, wr, 0.047)]
        wrists.append(wr)
    out.append(("hoodie", smin(torso, np.minimum.reduce(arms), 0.04)))
    legs, feet = [], []
    for side, kx in ((-1, -2.48), (1, -2.42)):
        knee = V(kx, 0.6, ZF + 0.11 * side)
        ankle = V(kx - 0.04, 0.1, ZF + 0.12 * side)
        legs += [capsule(p, hip + V(0, -0.03, 0.1 * side), knee, 0.075), capsule(p, knee, ankle, 0.055)]
        feet.append(ellipsoid(p, ankle + V(-0.07, -0.05, 0), V(0.11, 0.045, 0.05)))
    out.append(("pants", np.minimum.reduce(legs)))
    out.append(("shoe", np.minimum.reduce(feet)))
    # head: skull, jaw, nose and brow so the profile reads; hair as a cap over the back and top
    hc, tilt = head_frame()
    h = head_local(p, hc, tilt)
    skull = sphere(h, V(0, 0, 0), 0.095)
    jaw = ellipsoid(h, V(-0.035, -0.07, 0), V(0.07, 0.05, 0.065))
    nose = capsule(h, V(-0.09, 0.005, 0), V(-0.118, -0.03, 0), 0.016)
    brow = ellipsoid(h, V(-0.08, 0.032, 0), V(0.03, 0.02, 0.065))
    neck = capsule(p, chest + V(-0.04, 0.06, 0), hc + V(0.02, -0.09, 0), 0.045)
    hands = np.minimum.reduce([ellipsoid(p, w + V(-0.05, -0.01, 0), V(0.065, 0.025, 0.04)) for w in wrists])
    face = smin(smin(skull, jaw, 0.03), np.minimum(nose, brow), 0.01)
    out.append(("skin", np.minimum.reduce([face, neck, hands])))
    keep = np.minimum(np.maximum(-0.035 - h[:, 0], -0.045 - h[:, 1]), 0.045 - h[:, 1])   # short at the sides
    hair = np.maximum(ellipsoid(h, V(0.01, 0.03, 0), V(0.106, 0.112, 0.103)), keep)
    for cx_, cz_ in CURLS:                                          # curls on top
        cy_ = 0.03 + 0.112 * math.sqrt(max(1 - (cx_ / 0.106) ** 2 - (cz_ / 0.103) ** 2, 0))
        hair = np.minimum(hair, sphere(h, V(cx_, cy_ - 0.008, cz_), 0.03))
    out.append(("hair", hair))
    band = np.maximum(np.abs(np.hypot(h[:, 1], h[:, 2]) - 0.118) - 0.013,
                      np.maximum(np.abs(h[:, 0] - 0.005) - 0.02, -h[:, 1]))
    out.append(("phones", band))
    out.append(("cup", ellipsoid(h, V(0.005, -0.01, -0.112), V(0.05, 0.055, 0.03))))
    # the cat on the sill, a silhouette against the sky
    cat = V(CAT_X, WY0, BACK_Z - 0.06)
    p, p_room = cat + (p - cat) / CAT_S, p              # the cat is modelled at unit size, then scaled
    body = ellipsoid(p, cat + V(0, 0.13, 0), V(0.12, 0.14, 0.1))
    cathead = sphere(p, cat + V(0.02, 0.32, -0.02), 0.075)
    ears = np.minimum(capsule(p, cat + V(-0.02, 0.35, -0.02), cat + V(-0.035, 0.405, -0.02), 0.01),
                      capsule(p, cat + V(0.06, 0.35, -0.02), cat + V(0.075, 0.405, -0.02), 0.01))
    a = 0.7 + POSE["tail"]
    knee = cat + V(-0.26, 0.02, -0.04)
    tip = knee + 0.15 * V(-math.sin(a), math.cos(a), -0.25)
    tail = np.minimum(capsule(p, cat + V(-0.1, 0.03, 0.02), knee, 0.022), capsule(p, knee, tip, 0.019))
    out.append(("cat", CAT_S * smin(smin(body, cathead, 0.04), np.minimum(ears, tail), 0.02)))
    p = p_room
    wire = np.full(len(p), 9.0)
    for a, b in zip(FAIRY[:-1], FAIRY[1:]):
        wire = np.minimum(wire, capsule(p, a, b, 0.004))
    out.append(("wire", wire))
    # rug
    out.append(("rug", rbox(p, V(-0.9, 0.006, 1.6), V(1.3, 0.006, 0.95), 0.004)))
    # drum kit: bass drum, toms and snare; hi-hat, crash and ride on stands; a throne behind
    q = (p - DRUMS) / DRUM_S
    S = DRUM_S
    shells = [cyl_z(q, V(0, 0.27, 0), 0.25, 0.17)] + [cyl_y(q, c, r, hh) for c, r, hh in KIT_TOMS]
    out.append(("drum", S * np.minimum.reduce(shells)))
    out.append(("brass", S * np.minimum.reduce([disc(q, c, r, n) for c, r, n in KIT_CYMBALS])))
    stands = [capsule(q, V(c[0], 0, c[2]), c, 0.012) for c, _, _ in KIT_CYMBALS[1:]]
    stands += [capsule(q, V(-0.42, 0, 0.2), V(-0.42, 0.53, 0.2), 0.012),
               capsule(q, V(-0.02, 0.45, 0.05), V(-0.02, 0.58, 0.1), 0.012),
               cyl_y(q, V(0.02, 0.48, 0.55), 0.16, 0.035), capsule(q, V(0.02, 0, 0.55), V(0.02, 0.45, 0.55), 0.02)]
    for lx_ in (0.3, 0.6):
        stands.append(capsule(q, V(lx_, 0, 0.25), V(lx_, 0.3, 0.25), 0.012))
    out.append(("stand", S * np.minimum.reduce(stands)))
    # a pair of sticks left on the desk
    out.append(("sticks", np.minimum(capsule(p, V(-2.62, DESK_Y + 0.008, ZF + 0.28), V(-2.98, DESK_Y + 0.008, ZF + 0.12), 0.008),
                                     capsule(p, V(-2.6, DESK_Y + 0.024, ZF + 0.2), V(-2.97, DESK_Y + 0.008, ZF + 0.26), 0.008))))
    # pinboard on the right wall
    out.append(("cork", box(p, V(3.39, 1.55, 1.6), V(0.012, 0.42, 0.62))))
    return out


NAMES = [n for n, _ in parts(np.zeros((1, 3)))]
WHO = ["hoodie", "hair", "skin", "phones", "cup", "pants", "shoe"]


def sdf(p):
    return np.minimum.reduce([d for _, d in parts(p)])


def material(p):
    ds = parts(p)
    return np.argmin(np.stack([d for _, d in ds]), 0), parts.book_ids


# ================================================================ camera


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
MOON_COL = V(0.62, 0.68, 0.80)
LAMP_COL = V(1.0, 0.66, 0.36)
SCREEN_COL = V(0.62, 0.72, 0.58)


def through_window(p, l=TO_MOON):
    """1 where a ray from p towards l leaves through a pane, 0 where wall or bar blocks it."""
    lz = l[:, 2] if l.ndim == 2 else l[2]           # one direction, or one per point
    t = (WIN_Z - p[:, 2]) / np.where(np.abs(lz) < 1e-6, 1e-6, lz)
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
        return V(0.3, 0.2, 0.13)[None] * (0.88 + 0.12 * np.sin(x * 31 + np.sin(z * 7) * 3))[:, None]
    if name == "drum":                    # dark red shells, bone heads; a red ring on the bass drum's face
        q = (p - DRUMS) / DRUM_S
        face = np.abs(q[:, 2]) > 0.165
        rr = np.hypot(q[:, 0], q[:, 1] - 0.27)
        top = np.zeros(len(p), bool)
        for c_, r_, hh in KIT_TOMS:
            top |= (q[:, 1] > c_[1] + hh - 0.004) & (np.hypot(q[:, 0] - c_[0], q[:, 2] - c_[2]) < r_ - 0.004)
        c = np.where((face | top)[:, None], V(0.72, 0.68, 0.6)[None], V(0.36, 0.08, 0.08)[None])
        c[face & (np.abs(rr - 0.17) < 0.022)] = V(0.55, 0.07, 0.08)
        return c
    if name == "book":
        return BOOK_COL[book]
    if name == "rug":
        u = np.abs(x + 0.9) / 1.3
        v = np.abs(z - 1.6) / 0.95
        ring = np.maximum(u, v)
        band = (np.sin(ring * 30) > 0.3) & (ring > 0.78)
        diamond = ((np.abs(u * 4 % 1 - 0.5) + np.abs(v * 3 % 1 - 0.5)) < 0.3) & (ring < 0.7)
        return np.where((band | diamond)[:, None], V(0.4, 0.31, 0.2)[None], V(0.3, 0.09, 0.08)[None])
    if name == "cork":
        c = V(0.55, 0.4, 0.25)[None] * (0.85 + 0.15 * np.sin(y * 113) * np.sin(z * 97))[:, None]
        for yy, zz, col in ((1.75, 1.2, V(0.8, 0.72, 0.55)), (1.45, 1.75, V(0.55, 0.6, 0.38)),
                            (1.62, 2.05, V(0.9, 0.88, 0.82)), (1.32, 1.45, V(0.7, 0.3, 0.28))):
            c[(np.abs(y - yy) < 0.11) & (np.abs(z - zz) < 0.13)] = col
        return c
    flat = {"frame": V(0.5, 0.48, 0.44), "trim": V(0.5, 0.48, 0.44), "sill": V(0.55, 0.52, 0.47),
            "bezel": V(0.07, 0.07, 0.07), "screen": V(0.02, 0.02, 0.02), "keys": V(0.18, 0.18, 0.18),
            "lamp": V(0.5, 0.13, 0.11), "bulb": V(1, 1, 1), "mug": V(0.75, 0.73, 0.68),
            "board": V(0.12, 0.2, 0.1), "shelf": V(0.38, 0.26, 0.17), "pot": V(0.5, 0.38, 0.26),
            "leaf": V(0.3, 0.36, 0.16), "chair": V(0.22, 0.2, 0.2), "hoodie": V(0.55, 0.53, 0.48),
            "skin": V(0.8, 0.62, 0.5), "hair": V(0.56, 0.41, 0.24), "phones": V(0.15, 0.15, 0.15), "wire": V(0.1, 0.1, 0.1),
            "cup": V(0.6, 0.08, 0.08), "pants": V(0.32, 0.3, 0.27), "shoe": V(0.12, 0.12, 0.12),
            "cat": V(0.09, 0.09, 0.09), "brass": V(0.78, 0.64, 0.38), "stand": V(0.35, 0.35, 0.35),
            "sticks": V(0.72, 0.56, 0.36)}
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
    cone = np.clip((-(l @ LAMP_AIM) - 0.3) / 0.5, 0, 1)  # aimed at the face and hands, some spill
    sh = soft_shadow(q, l, ld - 0.06, k=10)
    col += alb * LAMP_COL * (0.4 * cone * sh * np.clip((n * l).sum(1), 0, 1) / (ld ** 2 + 0.02))[:, None]
    # monitor glow, only in front of the screen
    sv = MON_C + MON_N * 0.08 - q
    sd = length(sv)
    s = sv / sd[:, None]
    front = np.clip(-(s @ MON_N), 0, 1) ** 0.5
    sh2 = soft_shadow(q, s, sd - 0.1, k=6)
    col += alb * SCREEN_COL * (0.12 * front * sh2 * np.clip((n * s).sum(1), 0, 1) / (sd ** 2 + 0.05))[:, None]
    # fairy lights: small, warm, unshadowed
    for i, fp in enumerate(FAIRY):
        fv = fp + V(0, -0.02, -0.03) - q
        fd = length(fv)
        col += alb * FAIRY_COL[i % 4] * (0.0025 * np.clip((n * fv).sum(1) / fd, 0, 1) / (fd ** 2 + 0.01))[:, None]
    # ambient night-blue from the window
    sky = np.clip(0.5 + 0.5 * n[:, 2], 0, 1) * 0.07 + 0.02
    col += alb * (V(0.42, 0.46, 0.56) * sky[:, None]) * occl[:, None]
    name = np.array(NAMES)[mat]
    # a faint cold rim keeps the figure's outline off the dark wall
    who = np.isin(name, WHO)
    if who.any():
        v = CAM - p[who]
        v /= length(v)[:, None]
        rim = np.clip(1 - np.abs((n[who] * v).sum(1)), 0, 1) ** 2.5
        col[who] += MOON_COL * (0.12 * rim)[:, None]
    brass = name == "brass"                # cymbals mirror the moonlit window
    if brass.any():
        v = CAM - p[brass]
        v /= length(v)[:, None]
        nb = n[brass]
        r = -v + 2 * (nb * v).sum(1)[:, None] * nb
        r /= length(r)[:, None]
        w = WIN_AIM - p[brass]
        w /= length(w)[:, None]
        gloss = np.clip(((r * w).sum(1) - 0.55) / 0.35, 0, 1) ** 1.5     # a soft, brushed reflection
        col[brass] += alb[brass] * V(0.6, 0.62, 0.7) * (0.5 * gloss)[:, None]
    legs = np.isin(name, ["pants", "shoe", "drum", "stand"])  # the moonlit floor bounces light onto the legs and the kit
    if legs.any():
        col[legs] += alb[legs] * MOON_COL * (0.16 * np.clip(0.6 - 0.6 * n[legs, 1] - 0.4 * n[legs, 2], 0, 1))[:, None]
    cat = name == "cat"                    # the cat is back-lit: a moonlit outline against the sky
    if cat.any():
        v = CAM - p[cat]
        v /= length(v)[:, None]
        rim = np.clip(1 - np.abs((n[cat] * v).sum(1)), 0, 1) ** 4
        col[cat] += MOON_COL * (0.3 * rim)[:, None]
    col[name == "bulb"] = V(1.6, 1.15, 0.7) * 3
    col[name == "screen"] = V(0.015, 0.018, 0.014)
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
    for c, colr, k, r in ((LAMP_BULB, LAMP_COL, 0.06, 0.1),
                          (MON_C + MON_N * 0.05, SCREEN_COL, 0.008, 0.45)):
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


TURBINES = [(0.61, 1.0), (0.67, 1.3), (0.75, 1.15), (0.82, 0.95), (0.88, 0.8)]   # (u, size)


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
    # gradient: a dusky slate glow at the horizon, near-black overhead, and a thin warm band low down
    hor = np.exp(-np.clip(v, 0, None) / 0.07)
    col = V(0.005, 0.006, 0.011)[None] + V(0.10, 0.11, 0.15)[None] * (0.6 * hor)[:, None]
    col += V(0.30, 0.20, 0.12)[None] * (0.25 * np.exp(-((v - 0.035) / 0.022) ** 2))[:, None]
    # milky way
    band = np.exp(-((v - (0.62 - 0.5 * u)) / 0.05) ** 2)
    col += V(0.022, 0.022, 0.026)[None] * (band * (0.6 + 0.4 * np.sin(u * 90) * np.sin(v * 70)))[:, None]
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
    col += V(0.95, 0.93, 0.88)[None] * star[:, None]
    # the moon, a crescent with a halo
    mu, mv, mr = 0.68, 0.27, 0.04
    dm = np.hypot(u - mu, v - mv)
    lit = np.clip((mr - dm) / 0.003, 0, 1) * np.clip((np.hypot(u - mu + 0.024, v - mv - 0.015) - mr * 0.9) / 0.003, 0, 1)
    col += V(1.0, 0.95, 0.85)[None] * (3.0 * lit)[:, None]
    col += V(0.16, 0.16, 0.18)[None] * (0.35 * np.exp(-dm / 0.04))[:, None]
    # clouds, lit from above by the moon
    cl = np.zeros_like(u)
    for k, (f, a) in enumerate(((11, 1.0), (23, 0.5), (47, 0.25), (93, 0.12))):
        cl += a * np.sin(u * f + np.sin(v * f * 1.7 + k) + k * 3.1) * np.sin(v * f * 2.1 + u * f * 0.3 + k)
    layer = np.exp(-((v - 0.15) / 0.035) ** 2) + 0.7 * np.exp(-((v - 0.24) / 0.02) ** 2)
    dc = np.clip((cl * 0.6 + 0.25) * layer, 0, 1)
    edge = np.clip(dc * 3, 0, 1) * (1 - np.clip(dc * 2 - 0.4, 0, 1))
    cloud = V(0.018, 0.02, 0.026)[None] * (1 - dc[:, None]) + V(0.06, 0.06, 0.068)[None] * dc[:, None]
    col = col * (1 - dc[:, None]) + cloud * dc[:, None]
    col += V(0.55, 0.45, 0.34)[None] * (0.12 * edge * np.exp(-np.abs(u - mu) / 0.12))[:, None]
    # shooting stars, twice per loop
    for s0, s1, a0, dirn in ((18, 27, (0.6, 0.31), (0.12, -0.06)), (46, 53, (0.9, 0.28), (-0.1, -0.05))):
        if not s0 <= frame < s1:
            continue
        a0, dirn = np.array(a0), np.array(dirn)
        k = (frame - s0 + 1) / (s1 - s0)
        head = a0 + dirn * k
        dd, hh = seg_dist(u, v, head - dirn * 0.4, head)
        fade = math.sin(k * math.pi * 0.95)
        col += V(0.95, 0.93, 0.88)[None] * (6.0 * fade * hh ** 2 * np.exp(-(dd / 0.003) ** 2))[:, None]
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
    col[far] = col[far] * 0.12 + V(0.03, 0.032, 0.04)
    nh = hills(u, -0.045, 0.02, 5.0)
    near = v < nh
    col[near] = V(0.012, 0.012, 0.015)
    # lit windows on the near hill
    wu = np.floor(u / 0.009)
    wh = np.sin(wu * 91.7) * 9431.7
    wh -= np.floor(wh)
    win = near & (wh > 0.86) & (np.abs(v - (nh - 0.012 - 0.01 * wh)) < 0.0028) & (np.abs(u - (wu + 0.5) * 0.009) < 0.002)
    col[win] = V(1.0, 0.7, 0.35) * 1.2
    # two cars on the road along the near hill, one each way, once per loop
    for ph, sgn, c in ((t / loop, 1, V(1.0, 0.9, 0.7)), ((t / loop + 0.5) % 1, -1, V(1.0, 0.15, 0.08))):
        cu = 0.54 + 0.42 * (ph if sgn > 0 else 1 - ph)
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


def haze(mats):
    """How much of the scattered moonlight each ray picks up: little in front of
    the cat, so it stays a dark shape against the bright horizon."""
    return np.where(np.isin(mats, [i for i, n in enumerate(NAMES) if n == "cat"]), 0.15, 1.0)


def render_rays(d):
    """Colour along each ray (without the sky, which is added per frame) and whether it reached the sky."""
    t, hit = march(CAM, d)
    col = np.zeros((len(d), 3))
    h = np.nonzero(hit)[0]
    p = CAM + d[h] * t[h, None]
    mat, book = material(p)
    col[h] = shade(p, normals(p), mat, book)
    t_air = np.where(hit, t, (BACK_Z - CAM[2]) / np.maximum(d[:, 2], 1e-3))
    mats = np.full(len(d), -1)
    mats[h] = mat
    col += beam(CAM, d, np.minimum(t_air, 12)) * haze(mats)[:, None]
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
    mats = np.full(len(d), -1)
    mats[idx] = mat
    col += beam(CAM, d, np.minimum(t_air, 12)) * haze(mats)[:, None]
    return dict(col=col, sky=~hit, dirs=d, mats=mats)


def cells(a):
    """Average a picture (H, W) or (H, W, n) over the character cells -> (ROWS, COLS[, n])."""
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        return np.stack([cells(a[..., i]) for i in range(a.shape[2])], -1)
    return np.asarray(Image.fromarray(a, "F").resize((COLS, ROWS), Image.BOX), float)


DENSITY = " .'`,:;-~=+i!*co%x#&@"
_cov = Matcher(DENSITY).masks.mean(1)
DENS_CH = np.array(list(DENSITY))[np.argsort(_cov)]
DENS_COV = np.sort(_cov)                 # share of the cell each glyph inks, 0 .. 0.34


def blur(a, r=2):
    p = np.pad(a, r, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(2 * r + 1) for j in range(2 * r + 1)) / (2 * r + 1) ** 2


LUMA = np.array([0.3, 0.55, 0.15])


def to_cells(rgb, black=0.2, gain=2.2, expo=0.5, lo=0.1, gamma=0.8, sat=1.1):
    """Tone-mapped picture (H, W, 3) -> glyphs, the ink each is printed in, and the cell colours.

    A glyph covers at most a third of its cell, so print is much dimmer than
    the picture. To keep a night scene readable the range is spent where it
    matters: everything below `black` is left empty, and above it both the
    glyph (denser = lighter) and the ink (brighter, in the cell's own hue)
    rise with the cell's value."""
    T = cells(rgb)
    g = T @ LUMA
    T = np.clip(g[..., None] + sat * (T - g[..., None]), 0, None)
    m = T.max(-1)
    v = np.clip((m * gain - black) / (1 - black), 0, 1)
    ink = np.clip(T / np.maximum(m, 1e-6)[..., None] * (v ** expo)[..., None], 0, 1)
    want = DENS_COV[-1] * np.clip(lo + (1 - lo) * v ** gamma, 0, 1)
    ch = DENS_CH[np.abs(DENS_COV[None, None] - want[..., None]).argmin(-1)]
    ch[v < 0.012] = " "
    return ch, ink, T


def quantise(colours, k=96, iters=24):
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
    if not mask.any():
        return
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


# ---------------------------------------------------------------- the neon sign
# single-stroke letters on a unit square; each is a list of polylines
STROKES = {
    "A": [[(0, 0), (0.5, 1), (1, 0)], [(0.22, 0.42), (0.78, 0.42)]],
    "R": [[(0, 0), (0, 1), (0.7, 1), (1, 0.86), (1, 0.62), (0.7, 0.48), (0, 0.48)], [(0.45, 0.48), (1, 0)]],
    "D": [[(0, 0), (0, 1), (0.55, 1), (1, 0.72), (1, 0.28), (0.55, 0), (0, 0)]],
    "Ç": [[(1, 0.8), (0.72, 1), (0.3, 1), (0, 0.72), (0, 0.28), (0.3, 0), (0.72, 0), (1, 0.2)],
          [(0.5, 0), (0.5, -0.16), (0.3, -0.26)]],
    "B": [[(0, 0), (0, 1), (0.62, 1), (0.92, 0.87), (0.92, 0.63), (0.62, 0.5), (0, 0.5)],
          [(0.62, 0.5), (1, 0.37), (1, 0.13), (0.62, 0), (0, 0)]],
    "U": [[(0, 1), (0, 0.26), (0.3, 0), (0.7, 0), (1, 0.26), (1, 1)]],
    "K": [[(0, 0), (0, 1)], [(1, 1), (0, 0.44)], [(0.32, 0.62), (1, 0)]],
}
NEON_WORD = "ARDA ÇABUK"
NEON_X0, NEON_Y0 = -2.84, 2.45            # lower-left corner on the back wall
NEON_W, NEON_H, NEON_GAP, NEON_SPACE = 0.16, 0.32, 0.06, 0.16
NEON_COL = V(1.0, 0.10, 0.12)
# frame -> brightness; the sign hums, stutters once, and its Ç is a little loose
NEON_FLICKER = {21: 0.1, 22: 0.7, 23: 0.05, 24: 0.05, 25: 0.8, 53: 0.45}
C_FLICKER = {36: 0.0, 37: 1.0, 38: 0.0, 39: 0.0, 40: 0.6}


def neon_tubes():
    """Per letter, the light of its tube on the picture: (H, W, 3), tube core plus the glow on the wall."""
    out = []
    x = NEON_X0
    px, py = XX.ravel().astype(float), YY.ravel().astype(float)
    for ch in NEON_WORD:
        if ch == " ":
            x += NEON_SPACE
            continue
        d = np.full(H * W, 1e9)
        for line in STROKES[ch]:
            pts = [project(V(x + u * NEON_W, NEON_Y0 + v * NEON_H, BACK_Z - 0.02)) for u, v in line]
            for a, b in zip(pts[:-1], pts[1:]):
                d = np.minimum(d, seg_dist(px, py, a, b)[0])
        core = np.exp(-(d / 1.1) ** 2)
        glow = 0.1 * np.exp(-(d / 4.0) ** 2) + 0.03 * np.exp(-(d / 18) ** 2)
        out.append((core[:, None] * V(2.6, 0.9, 0.8) + glow[:, None] * NEON_COL).astype(np.float32))
        x += NEON_W + NEON_GAP
    return out


def neon_level(frame, letter):
    b = NEON_FLICKER.get(frame, 1.0)
    if letter == 4:                        # the Ç
        b *= C_FLICKER.get(frame, 1.0)
    return b


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
LED = BOARD + V(0.04, 0.012, 0.08)


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
        lights.append((LED, V(1.0, 0.15, 0.12) * 1.4, 0.8))
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
    who = np.isin(img, [NAMES.index(n) for n in WHO])
    pidx = np.nonzero((blur(who.astype(float), 12) > 0).ravel())[0]
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

    tubes = neon_tubes()
    pics = []
    for f in range(FRAMES):
        slow = f - f % SLOW                    # twinkles move on a slower clock
        c, sk = col.copy(), sky.copy()
        c[pidx], sk[pidx] = PC[f % 8], PS[f % 8]
        c[cidx], sk[cidx] = CC[(f // SLOW) % TAILS], CS[(f // SLOW) % TAILS]
        c[sk] += sky_colour(dirs[sk], f, slow)
        c += glows(f, slow)
        for i, tube in enumerate(tubes):
            c += tube * neon_level(f, i)
        ch, ink, P = to_cells(tonemap(c).reshape(H, W, 3))
        if f:                                   # keep cells whose target barely moved as they were
            same = np.abs(P - pics[0][2]).max(-1) < 0.02
            ch[same], ink[same] = pics[0][0][same], pics[0][1][same]
        pics.append((ch, ink, P))

    cent = quantise(np.concatenate([ink[ch != " "] for ch, ink, _ in pics[::8]]))
    names = [f"{'cdefghijlmnopq'[i // 10]}{i % 10}" for i in range(len(cent))]
    palette = {n: "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c) for n, c in zip(names, cent)}
    palette.update({"sc": "#e6e2d6", "sd": "#aba99f", "sb": "#5d6048", "st": "#77776f"})
    names = np.array(names)

    img = mats.reshape(H, W)
    screen = cells(img == NAMES.index("screen")) > 0.6
    mug = np.argwhere(cells(img == NAMES.index("mug")) > 0.3)
    mr, mc = mug[:, 0].min(), int(round(mug[:, 1].mean()))

    grids = []
    for f, (ch, ink, _) in enumerate(pics):
        cl = names[((ink[..., None, :] - cent[None, None]) ** 2).sum(-1).argmin(-1)].astype("<U2")
        ch = ch.copy()
        monitor(ch, cl, screen, f)
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

    movie = Movie(COLS, ROWS, pad=(12, 14), palette=palette, bold=True, font_px=17,
                  title="Arda Çabuk — embedded software and machine learning",
                  desc="A ray-marched room at night, drawn in ASCII. A red neon sign on the back wall reads ARDA ÇABUK "
                       "and stutters now and then. In the foreground someone with short curly dark-blond hair, a pale hoodie and "
                       "headphones with a red ear cup sits in profile at a desk against the left wall, lit by an amber "
                       "desk lamp, nodding to the music and typing; the monitor reads arda@jetson and shows a scrolling "
                       "vibration trace marked nominal, a development board blinks beside the keyboard, steam rises "
                       "from a mug and a pair of drumsticks lies on the desk. Between the desk and the window stands a drum kit, "
                       "its cymbals catching the moonlight. Through a tall arched window a cat sits on the sill against the horizon, a crescent "
                       "moon hangs in a slate sky, stars twinkle, shooting stars cross, cars pass on the hill road and "
                       "wind turbines turn on the ridge with red lights blinking. Fairy lights twinkle under the ceiling "
                       "and dust drifts in the moonlight falling through the window.")
    movie.add(static)
    for fr, hold in sprites:
        movie.add(fr, FPS, hold=hold)
    return movie, pics


if __name__ == "__main__":
    m, _ = build()
    if "--preview" in sys.argv:
        m.preview("hero.png", times=(0, 2.5, 5.0))
    m.write("hero.svg")
