"""Hero, take two: someone at a drum kit, and every hit throws code into the air.

The same machinery as hero.py: a 3-D scene of signed distance functions,
ray-marched in numpy and printed in ASCII with a quantised full-colour palette.
The drummer (short curly dark-blond hair, a pale hoodie, red headphones) sits
behind a red kit in a dark practice room, lit by one warm spotlight from above
with its cone hanging in the haze and a red light behind.

The room is rendered once. The figure is re-rendered for every pose the groove
needs (each hand is a target drum and a stick lift; the head nods on the kick),
and each hit flashes the drum it lands on (a point light just above its head).
On top of the picture, every hit launches code: words and symbols fly up out of
the drum, arc under gravity and fade from bone to olive to nothing. A crash
throws a fountain, the snare a few keywords, the ride a single symbol, the kick
lets bits drift up off the bass drum, whose front head carries the name.

The groove is four bars of 4/4 at 120 bpm, eighth notes, 8 frames per second:
three bars of ride, snare and kick, then a fill around the toms back to the crash.
"""
import math
import pathlib
import sys

import numpy as np
from PIL import Image

from engine import NBSP, Grid, Matcher, Movie, pack
from hero import (V, length, box, rbox, sphere, capsule, cyl_y, ellipsoid, smin, tonemap, quantise,
                  DENS_CH, DENS_COV, LUMA, blur)

QUICK = "--quick" in sys.argv
W, H = (400, 224) if QUICK else (800, 448)
COLS, ROWS = 240, 67
FPS, FRAMES = 8, 64
EXPO = 0.45                             # the room is lit; print only what the light picks out
STEP = 2                                # frames per eighth note

CACHE = pathlib.Path(__file__).with_name("_preview")


# ================================================================ extra sdf pieces
def cyl_n(p, c, n, r, hh):
    """A cylinder centred at c with axis n (unit), radius r, half height hh."""
    q = p - c
    a = q @ n
    rad = length(q - a[:, None] * n[None])
    d = np.stack([rad - r, np.abs(a) - hh], -1)
    return np.minimum(d.max(-1), 0) + length(np.maximum(d, 0))


def disc(p, c, r, n, t=0.006):
    q = p - c
    a = q @ n
    rad = length(q - a[:, None] * n[None])
    # a cymbal: thin, slightly domed towards the bell
    d = np.stack([rad - r, np.abs(a - 0.03 * np.clip(1 - rad / r, 0, 1) ** 2) - t], -1)
    return np.minimum(d.max(-1), 0) + length(np.maximum(d, 0))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ================================================================ camera
# two takes of the same room: from the front (the default) or from the drummer's
# right side (--side), where the arms and sticks stand clear of the toms
SIDE = "--side" in sys.argv
VIEW = "side" if SIDE else "front"
CAM = V(-2.0, 1.5, -1.45) if SIDE else V(0.6, 1.62, -2.05)
LOOK = V(0.05, 0.95, 0.42) if SIDE else V(0.08, 0.92, 0.5)
HFOV = math.radians(56)
BACK_Z = 2.1


# ================================================================ the kit
# the drummer faces -z (the camera); their right hand is on -x, the left on +x.
# drums: (name, centre, axis, radius, half height); the head is the face along +axis
UP = V(0, 1, 0)
DRUMS = [
    ("snare", V(0.34, 0.66, 0.4), unit((-0.08, 1, 0.18)), 0.18, 0.065),
    ("tom1", V(0.15, 0.76, 0.28), unit((0.1, 1, 0.45)), 0.13, 0.085),
    ("tom2", V(-0.16, 0.77, 0.3), unit((-0.1, 1, 0.45)), 0.14, 0.09),
    ("floor", V(-0.47, 0.48, 0.5), unit((0.05, 1, 0.08)), 0.2, 0.19),
]
KICK_C, KICK_R, KICK_H = V(0.0, 0.29, 0.1), 0.29, 0.2           # bass drum lies along z, front head at -z
CYMBALS = [
    ("hihat", V(0.6, 0.92, 0.48), unit((0, 1, 0.05)), 0.18),
    ("crash", V(0.56, 1.3, 0.3), unit((-0.15, 1, 0.2)), 0.21),
    ("ride", V(-0.72, 1.02, 0.8), unit((0.25, 1, 0.0)), 0.24) if SIDE else
    ("ride", V(-0.64, 1.04, 0.36), unit((0.2, 1, 0.15)), 0.24),
]
CYM = {n: (c, ax, r) for n, c, ax, r in CYMBALS}
DRUM = {n: (c, ax, r, hh) for n, c, ax, r, hh in DRUMS}


def head_point(name):
    """Where a stick lands on a drum or cymbal (and the flash comes from)."""
    if name in DRUM:
        c, ax, r, hh = DRUM[name]
        toward = unit(V(0, 0, 0.75) - c) * 0.35 * r            # a little off centre, towards the player
        toward -= (toward @ ax) * ax
        return c + ax * hh + toward
    if name == "kick":
        return KICK_C + V(0, 0.12, -KICK_H)          # the upper half of the front head is in frame
    c, ax, r = CYM[name]
    edge = unit(V(0, 1.0, 0.75) - c)                           # the edge nearest the player
    edge -= (edge @ ax) * ax
    return c + unit(edge) * 0.72 * r + ax * 0.01


TARGET = {n: head_point(n) for n in ("snare", "tom1", "tom2", "floor", "hihat", "crash", "ride")}

# ================================================================ the groove
# (step, limb, target); a step is an eighth note, 32 steps = 4 bars
HITS = [(0, "L", "crash")]
for bar in range(3):
    s = bar * 8
    HITS += [(s + k, "R", "ride") for k in (0, 2, 4, 6)]
    HITS += [(s + k, "L", "snare") for k in (2, 6)]
    HITS += [(s + k, "K", "kick") for k in ((0, 4, 5) if bar != 1 else (0, 3, 4))]
HITS = [h for h in HITS if not (h[0] == 0 and h[1] == "R")] + [(0, "R", "ride")]
HITS += [(24, "R", "tom2"), (25, "L", "tom1"), (26, "R", "tom2"), (27, "L", "tom1"),
         (28, "R", "floor"), (29, "L", "snare"), (30, "R", "floor"), (31, "L", "snare"),
         (24, "K", "kick"), (28, "K", "kick")]
HITS = sorted(set(HITS))
LIMB_HITS = {l: sorted((s * STEP, t) for s, l_, t in HITS if l_ == l) for l in "LRK"}


def hand_state(limb, f):
    """(target, lift) of one hand at frame f: lift 0 = stick on the head, 1 = raised."""
    hs = LIMB_HITS[limb]
    fr = [h for h, _ in hs]
    i = max(k for k in range(len(hs)) if fr[k] <= f) if fr[0] <= f else -1
    h0, t0 = hs[i] if i >= 0 else (hs[-1][0] - FRAMES, hs[-1][1])
    h1, t1 = hs[i + 1] if i + 1 < len(hs) else (hs[0][0] + FRAMES, hs[0][1])
    dt, rem = f - h0, h1 - f
    if dt == 0:
        return t0, 0.0
    if dt == 1 and rem == 1:
        return t1, 0.5
    if dt == 1:
        return t0, 0.55
    if rem == 1:
        return t1, 0.6
    return t1, 1.0


def kick_on(f):
    return any(h == f for h, _ in LIMB_HITS["K"])


def pose_at(f):
    return (*hand_state("R", f), *hand_state("L", f), 1.0 if kick_on(f) else 0.0)


POSE = {"key": pose_at(1)}


# ================================================================ the drummer
HIP, CHEST = V(0, 0.62, 0.8), V(0, 1.0, 0.74)
SHOULDER = {"R": CHEST + V(-0.19, 0.0, 0.0), "L": CHEST + V(0.19, 0.0, 0.0)}
STICK = 0.36
CURLS = [(x_, z_) for x_ in (-0.06, -0.02, 0.02, 0.06) for z_ in (-0.04, 0.0, 0.04, 0.08)]


def arm(limb, target, lift):
    """Shoulder, elbow, wrist and the stick (grip -> tip) for a hand aiming at a target."""
    S = SHOULDER[limb]
    T = TARGET[target]
    rest = S + V(0, -0.32, -0.2)                         # where a relaxed hand sits
    sd = unit(unit(T - rest) + V(0, -0.25, -0.9))        # sticks point away from the body
    grip_d = T - STICK * sd
    if grip_d[2] > 0.5:                                  # hands stay in front of the belly
        grip_d = grip_d + V(0, 0, 0.5 - grip_d[2])
        sd = unit(T - grip_d)
    perp = unit(UP - (UP @ sd) * sd)
    a = lift * 1.15
    s = unit(sd * math.cos(a) + perp * math.sin(a))
    grip = grip_d + lift * V(0, 0.11, 0.04)
    tip = grip + STICK * s
    wrist = grip - 0.035 * s + V(0, -0.01, 0.02)
    # two-bone reach for the elbow, bending out and down
    up_len, fore = 0.29, 0.27
    d = wrist - S
    L = min(np.linalg.norm(d), up_len + fore - 1e-3)
    dn = d / np.linalg.norm(d)
    along = (up_len ** 2 - fore ** 2 + L ** 2) / (2 * L)
    hgt = math.sqrt(max(up_len ** 2 - along ** 2, 0))
    side = -1 if limb == "R" else 1
    pole = V(side * 0.8, -1.0, 0.45)
    pole = unit(pole - (pole @ dn) * dn)
    # swing the elbow around the shoulder-wrist line and keep the bend that clears
    # the torso by the arm's thickness, closest to the natural out-and-down pole
    u2 = np.cross(dn, pole)
    best = None
    for a in np.linspace(-math.pi, math.pi, 49):
        e = S + dn * along + (pole * math.cos(a) + u2 * math.sin(a)) * hgt
        pts = np.array([S + (e - S) * k for k in (0.5, 0.75, 1.0)] + [e + (wrist - e) * k for k in (0.25, 0.5, 0.75)])
        clear = (torso(pts) - 0.065).min()
        score = min(clear, 0.0) * 50 + math.cos(a)
        if best is None or score > best[0]:
            best = (score, e)
    elbow = best[1]
    return S, elbow, wrist, grip, tip


def torso(p):
    return smin(capsule(p, HIP + V(-0.07, 0, 0), CHEST + V(-0.08, 0, 0), 0.13),
                capsule(p, HIP + V(0.07, 0, 0), CHEST + V(0.08, 0, 0), 0.13), 0.05)


def head_frame():
    nod = POSE["key"][4]
    return V(0, 1.3 - 0.022 * nod, 0.68 - 0.015 * nod), 0.16 * nod


def head_local(p, c, tilt):
    """Relative to the head, un-tilted: -z is the face, y is up."""
    q = p - c
    cs, sn = math.cos(tilt), math.sin(tilt)
    return np.stack([q[:, 0], cs * q[:, 1] - sn * q[:, 2], sn * q[:, 1] + cs * q[:, 2]], -1)


def parts(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    out = []
    out.append(("floor", y))
    out.append(("wall", BACK_Z - z))
    out.append(("wall", x + 3.2))
    out.append(("wall", 3.2 - x))
    out.append(("led", box(p, V(0, 0.035, BACK_Z - 0.01), V(3.0, 0.008, 0.01))))
    out.append(("rug", rbox(p, V(0.0, 0.006, 0.42), V(1.15, 0.006, 0.95), 0.004)))
    # the amp to the drummer's left and a cat on it
    out.append(("amp", rbox(p, V(1.42, 0.36, 0.95), V(0.3, 0.36, 0.2), 0.02)))
    cat = V(1.38, 0.72, 0.95)
    body = ellipsoid(p, cat + V(0, 0.13, 0), V(0.1, 0.13, 0.11))
    chead = sphere(p, cat + V(-0.02, 0.29, -0.03), 0.07)
    ears = np.minimum(capsule(p, cat + V(-0.06, 0.32, -0.03), cat + V(-0.075, 0.375, -0.03), 0.01),
                      capsule(p, cat + V(0.02, 0.32, -0.03), cat + V(0.035, 0.375, -0.03), 0.01))
    tail = np.minimum(capsule(p, cat + V(0.05, 0.02, 0.08), cat + V(0.19, 0.02, 0.02), 0.02),
                      capsule(p, cat + V(0.19, 0.02, 0.02), cat + V(0.25, 0.1, -0.05), 0.018))
    out.append(("cat", smin(smin(body, chead, 0.04), np.minimum(ears, tail), 0.02)))
    # kit: shells (with heads and hoops told apart by the albedo), cymbals and stands
    shells = [np.maximum(np.hypot(x - KICK_C[0], y - KICK_C[1]) - KICK_R, np.abs(z - KICK_C[2]) - KICK_H)]
    shells += [cyl_n(p, c, ax, r, hh) for _, c, ax, r, hh in DRUMS]
    S = np.stack(shells)
    parts.drum_ids = S.argmin(0)
    out.append(("drum", S.min(0)))
    out.append(("brass", np.minimum.reduce([disc(p, c, r, ax) for _, c, ax, r in CYMBALS]
                                           + [disc(p, CYM["hihat"][0] + V(0, 0.025, 0), 0.18, CYM["hihat"][1])])))
    st = []
    for _, c, ax, r in CYMBALS:
        foot = V(c[0] + 0.05 * np.sign(c[0]), 0, c[2] + 0.05)
        st.append(capsule(p, foot, c - ax * 0.01, 0.011))
        for k in range(3):
            a = k * 2.1 + 0.4
            st.append(capsule(p, foot + V(0, 0.32, 0), foot + V(0.17 * math.cos(a), 0, 0.17 * math.sin(a)), 0.009))
    st.append(capsule(p, V(0.34, 0, 0.4), DRUM["snare"][0] - V(0, 0.06, 0), 0.012))
    for k in range(3):
        a = k * 2.1 + 1.0
        st.append(capsule(p, V(0.34, 0.3, 0.4), V(0.34 + 0.18 * math.cos(a), 0, 0.4 + 0.18 * math.sin(a)), 0.009))
    for n_ in ("tom1", "tom2"):
        st.append(capsule(p, KICK_C + V(0.0, KICK_R - 0.02, 0.05), DRUM[n_][0] - DRUM[n_][1] * 0.08, 0.012))
    for lx in (-0.62, -0.32):
        st.append(capsule(p, V(lx, 0, 0.5), V(lx, 0.32, 0.5), 0.011))
    for sx_ in (-1, 1):                                               # bass drum spurs
        st.append(capsule(p, KICK_C + V(0.26 * sx_, -0.12, -0.12), V(0.36 * sx_, 0.0, -0.2), 0.01))
    st.append(box(p, V(-0.07, 0.02, 0.38), V(0.05, 0.012, 0.11)))    # kick pedal
    st.append(box(p, V(0.6, 0.02, 0.6), V(0.05, 0.012, 0.11)))      # hi-hat pedal
    st.append(cyl_y(p, V(0, 0.47, 0.84), 0.18, 0.045))               # throne
    st.append(capsule(p, V(0, 0.0, 0.84), V(0, 0.44, 0.84), 0.025))
    out.append(("stand", np.minimum.reduce(st)))
    # the drummer
    torso = smin(capsule(p, HIP + V(-0.07, 0, 0), CHEST + V(-0.08, 0, 0), 0.13),
                 capsule(p, HIP + V(0.07, 0, 0), CHEST + V(0.08, 0, 0), 0.13), 0.05)
    torso = smin(torso, capsule(p, CHEST + V(-0.16, 0, 0), CHEST + V(0.16, 0, 0), 0.085), 0.06)
    torso = smin(torso, ellipsoid(p, CHEST + V(0, 0.07, 0.09), V(0.12, 0.06, 0.08)), 0.03)    # the hood
    tR, lR, tL, lL, _ = POSE["key"]
    limbs, hands, sticks = [], [], []
    for limb, t_, l_ in (("R", tR, lR), ("L", tL, lL)):
        S_, E_, W_, G_, T_ = arm(limb, t_, l_)
        limbs += [capsule(p, S_, E_, 0.058), capsule(p, E_, W_, 0.045)]
        hands.append(sphere(p, G_ + V(0, 0.0, 0.01), 0.042))
        sticks.append(capsule(p, G_ - 0.07 * unit(T_ - G_), T_, 0.013))
    out.append(("hoodie", smin(torso, np.minimum.reduce(limbs), 0.04)))
    legs, feet = [], []
    for side, ank in ((-1, V(-0.08, 0.1, 0.44)), (1, V(0.5, 0.1, 0.56))):
        knee = V(0.16 * side, 0.6, 0.48) if side < 0 else V(0.3, 0.6, 0.55)
        legs += [capsule(p, HIP + V(0.1 * side, -0.03, 0), knee, 0.075), capsule(p, knee, ank, 0.055)]
        feet.append(ellipsoid(p, ank + V(0, -0.055, -0.07), V(0.05, 0.045, 0.11)))
    out.append(("pants", np.minimum.reduce(legs)))
    out.append(("shoe", np.minimum.reduce(feet)))
    hc, tilt = head_frame()
    h = head_local(p, hc, tilt)
    skull = sphere(h, V(0, 0, 0), 0.095)
    jaw = ellipsoid(h, V(0, -0.068, -0.03), V(0.068, 0.05, 0.07))
    nose = capsule(h, V(0, 0.005, -0.09), V(0, -0.03, -0.116), 0.016)
    brow = ellipsoid(h, V(0, 0.032, -0.08), V(0.066, 0.02, 0.03))
    neck = capsule(p, CHEST + V(0, 0.06, -0.02), hc + V(0, -0.09, 0.02), 0.046)
    face = smin(smin(skull, jaw, 0.03), np.minimum(nose, brow), 0.01)
    out.append(("skin", np.minimum.reduce([face, neck] + hands)))
    keep = np.minimum(np.maximum(-0.035 - h[:, 2], -0.045 - h[:, 1]), 0.05 - h[:, 1])   # off the face, short at the sides
    hair = np.maximum(ellipsoid(h, V(0, 0.03, 0.01), V(0.104, 0.112, 0.106)), keep)
    for cx_, cz_ in CURLS:
        cy_ = 0.03 + 0.112 * math.sqrt(max(1 - (cx_ / 0.104) ** 2 - ((cz_ - 0.01) / 0.106) ** 2, 0))
        hair = np.minimum(hair, sphere(h, V(cx_, cy_ - 0.008, cz_), 0.03))
    out.append(("hair", hair))
    band = np.maximum(np.abs(np.hypot(h[:, 0], h[:, 1]) - 0.12) - 0.012,
                      np.maximum(np.abs(h[:, 2] - 0.01) - 0.02, -h[:, 1]))
    out.append(("phones", band))
    out.append(("cup", np.minimum(ellipsoid(h, V(-0.11, -0.01, 0.01), V(0.03, 0.055, 0.05)),
                                  ellipsoid(h, V(0.11, -0.01, 0.01), V(0.03, 0.055, 0.05)))))
    out.append(("sticks", np.minimum.reduce(sticks)))
    return out


NAMES = [n for n, _ in parts(np.zeros((1, 3)))]
WHO = ["hoodie", "hair", "skin", "phones", "cup", "pants", "shoe", "sticks"]


def sdf(p):
    return np.minimum.reduce([d for _, d in parts(p)])


def material(p):
    ds = parts(p)
    return np.argmin(np.stack([d for _, d in ds]), 0), parts.drum_ids


def camera_rays(w=W, h=H):
    f = unit(LOOK - CAM)
    r = unit(np.cross(UP, f))
    u = np.cross(f, r)
    tx = math.tan(HFOV / 2)
    xs = ((np.arange(w) + 0.5) / w * 2 - 1) * tx
    ys = -((np.arange(h) + 0.5) / h * 2 - 1) * tx * h / w
    X, Y = np.meshgrid(xs, ys)
    d = f[None] + X.reshape(-1, 1) * r[None] + Y.reshape(-1, 1) * u[None]
    return d / length(d)[:, None]


def project(P):
    """3-D point -> (col, row) in character cells."""
    f = unit(LOOK - CAM)
    r = unit(np.cross(UP, f))
    u = np.cross(f, r)
    v = P - CAM
    zz = v @ f
    tx = math.tan(HFOV / 2)
    x = (v @ r / zz / tx + 1) / 2
    y = (-(v @ u) / zz / (tx * H / W) + 1) / 2
    return x * COLS, y * ROWS


def march(o, d, tmax=12.0, steps=200):
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
        out = t[idx] > tmax
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


def soft_shadow(p, l, tmax, k=12.0, steps=40):
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
        t[idx] += np.clip(h, 0.01, 0.25)
        alive[idx[(res[idx] < 0.01) | (t[idx] > tmax[idx])]] = False
    return res ** 1.5


def ao(p, n):
    occ, w = 0.0, 1.0
    for i in range(1, 6):
        hh = 0.03 * i
        occ = occ + w * np.clip(hh - sdf(p + n * hh), 0, None)
        w *= 0.6
    return np.clip(1 - 3.2 * occ, 0, 1)


# ================================================================ lights
KEY = V(-0.5, 3.0, -0.8) if SIDE else V(0.35, 3.0, -0.9)                 # the spotlight, above and a little in front
KEY_AIM = unit(V(0.02, 0.95, 0.55) - KEY)
KEY_COL = V(1.0, 0.86, 0.66)
RIM = V(-0.35, 2.2, 1.85)                # red, behind the drummer
RIM_COL = V(1.0, 0.09, 0.1)
FLASH_COL = V(1.0, 0.95, 0.86)


def key_cone(q):
    l = KEY - q
    l = l / length(l)[:, None]
    return np.clip((-(l @ KEY_AIM) - 0.86) / 0.08, 0, 1) ** 1.5


def albedo(name, p, did):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    c = np.ones((len(p), 3))
    if name == "floor":
        plank = np.floor(x / 0.19)
        seam = np.clip(np.abs((x / 0.19) % 1 - 0.5) * 2 - 0.9, 0, 1) * 6
        grain = 0.85 + 0.15 * np.sin(z * 9 + plank * 3.1 + np.sin(z * 2.3 + plank) * 2)
        tone = 0.75 + 0.25 * np.sin(plank * 12.9898) ** 2
        return V(0.26, 0.17, 0.11)[None] * (grain * tone * (1 - 0.6 * seam))[:, None]
    if name == "wall":                    # acoustic foam panels: a wedge pattern in squares
        u, v = (x + 3) / 0.5, (y - 0.3) / 0.5
        panel = (np.abs(u % 1 - 0.5) < 0.46) & (np.abs(v % 1 - 0.5) < 0.46) & (y > 0.3) & (y < 2.8)
        flip = (np.floor(u) + np.floor(v)) % 2 == 0
        wedge = np.where(flip, np.sin(u * 2 * math.pi * 4), np.sin(v * 2 * math.pi * 4))
        foam = (0.07 + 0.025 * wedge) if SIDE else (0.16 + 0.05 * wedge)
        return np.where(panel[:, None], foam[:, None] * V(0.9, 0.88, 0.85)[None], V(0.12, 0.11, 0.11)[None] if SIDE else V(0.3, 0.28, 0.27)[None])
    if name == "rug":
        u = np.abs(x) / 1.15
        v = np.abs(z - 0.42) / 0.95
        ring = np.maximum(u, v)
        band = (np.sin(ring * 34) > 0.3) & (ring > 0.8)
        return np.where(band[:, None], V(0.33, 0.36, 0.2)[None], V(0.16, 0.17, 0.1)[None])
    if name == "drum":                    # red shells, bone heads, chrome hoops; a red ring on the kick's face
        c = np.tile(V(0.42, 0.06, 0.07), (len(p), 1))
        spark = np.sin(x * 211 + y * 173) * np.sin(z * 197 - x * 89)
        c *= (0.85 + 0.15 * spark)[:, None]
        k = did == 0
        if k.any():
            q = p[k] - KICK_C
            rr = np.hypot(q[:, 0], q[:, 1])
            face = np.abs(q[:, 2]) > KICK_H - 0.004
            hoop = (rr > KICK_R - 0.025) & (np.abs(np.abs(q[:, 2]) - KICK_H) < 0.04)
            cc = c[k]
            cc[face] = V(0.74, 0.7, 0.62)
            cc[face & (np.abs(rr - 0.21) < 0.018)] = V(0.5, 0.05, 0.06)
            cc[hoop & ~face] = V(0.5, 0.5, 0.5)
            c[k] = cc
        for i, (_, dc, ax, r, hh) in enumerate(DRUMS):
            m = did == i + 1
            if not m.any():
                continue
            q = p[m] - dc
            a = q @ ax
            rad = length(q - a[:, None] * ax[None])
            cc = c[m]
            cc[(a > hh - 0.004) & (rad < r - 0.006)] = V(0.74, 0.7, 0.62)
            cc[(np.abs(a) > hh - 0.025) & (rad > r - 0.008)] = V(0.55, 0.55, 0.55)
            c[m] = cc
        return c
    if name == "amp":
        grille = (np.abs(z - 0.75) < 0.01) & (np.abs(x - 1.42) < 0.26) & (np.abs(y - 0.4) < 0.26)
        mesh = 0.8 + 0.2 * np.sin(x * 300) * np.sin(y * 300)
        return np.where(grille[:, None], (0.09 * mesh)[:, None] * V(1, 1, 1)[None], V(0.13, 0.12, 0.12)[None])
    if name == "skin":
        hc, tilt = head_frame()
        h = head_local(p, hc, tilt)
        eye = (h[:, 2] < -0.07) & (np.abs(np.abs(h[:, 0]) - 0.033) < 0.009) & (np.abs(h[:, 1] - 0.01) < 0.006)
        mouth = (h[:, 2] < -0.08) & (np.abs(h[:, 0]) < 0.022) & (np.abs(h[:, 1] + 0.058) < 0.005)
        c = np.tile(V(0.8, 0.62, 0.5), (len(p), 1))
        c[eye] = V(0.12, 0.09, 0.08)
        c[mouth] = V(0.45, 0.25, 0.22)
        return c
    flat = {"led": V(1, 1, 1), "cat": V(0.08, 0.08, 0.08), "brass": V(0.55, 0.42, 0.22),
            "stand": V(0.42, 0.42, 0.42), "hoodie": V(0.56, 0.54, 0.49), "hair": V(0.5, 0.36, 0.2),
            "phones": V(0.14, 0.14, 0.14), "cup": V(0.62, 0.07, 0.08), "pants": V(0.2, 0.2, 0.19),
            "shoe": V(0.1, 0.1, 0.1), "sticks": V(0.8, 0.64, 0.42)}
    return c * flat[name]


def shade(p, n, mat, did):
    """Radiance towards the camera without the hit flashes, plus the albedo they need."""
    col = np.zeros((len(p), 3))
    alb = np.zeros((len(p), 3))
    for i, name in enumerate(NAMES):
        m = mat == i
        if m.any():
            alb[m] = albedo(name, p[m], did[m])
    occl = ao(p, n)
    q = p + n * 2e-3
    name = np.array(NAMES)[mat]
    # the spotlight
    lv = KEY - q
    ld = length(lv)
    l = lv / ld[:, None]
    cone = key_cone(q)
    lit = cone > 0
    vis = np.zeros(len(p))
    if lit.any():
        vis[lit] = soft_shadow(q[lit], l[lit], ld[lit] - 0.1, k=14)
    col += alb * KEY_COL * (9.0 * cone * vis * np.clip((n * l).sum(1), 0, 1) / ld ** 2)[:, None]
    # red light behind
    rv = RIM - q
    rd = length(rv)
    r_ = rv / rd[:, None]
    sh = soft_shadow(q, r_, rd - 0.1, k=8, steps=32)
    col += alb * RIM_COL * (np.where(name == "wall", 0.08 if SIDE else 0.6, 2.2) * sh * np.clip((n * r_).sum(1), 0, 1) / (rd ** 2 + 0.1))[:, None]
    # a faint warm bounce off the lit rug and a cold fill so nothing is pure black
    col += alb * V(0.3, 0.24, 0.18) * (0.05 * np.clip(-n[:, 1] * 0.5 + 0.5, 0, 1) * occl)[:, None]
    col += alb * V(0.35, 0.38, 0.45) * (0.035 * occl)[:, None]
    # rims: a red edge on the figure and the cat, from the light behind
    v = CAM - p
    v /= length(v)[:, None]
    rim = np.clip(1 - np.abs((n * v).sum(1)), 0, 1) ** 3
    edge = np.isin(name, WHO + ["cat", "drum"])
    col[edge] += RIM_COL * (0.35 * rim[edge] * np.clip(0.4 + (n[edge] * r_[edge]).sum(1), 0, 1))[:, None]
    # cymbals: a soft highlight of the spotlight
    brass = name == "brass"
    if brass.any():
        nb = n[brass]
        rr = -v[brass] + 2 * (nb * v[brass]).sum(1)[:, None] * nb
        gloss = np.clip(((rr * l[brass]).sum(1) - 0.7) / 0.3, 0, 1) ** 2
        col[brass] += alb[brass] * KEY_COL * (0.9 * gloss * vis[brass] + 0.04)[:, None]
    col[name == "led"] = RIM_COL * 2.5
    return col, alb


def haze(o, d, t_end, n=40):
    """The spotlight's cone in the air, and a red glow around the light behind."""
    acc = np.zeros(len(d))
    dt = t_end / n
    for i in range(n):
        s = (i + 0.5) * dt
        p = o + d * s[:, None]
        acc += key_cone(p) * (p[:, 1] < 3.1) * dt
    light = KEY_COL[None] * ((0.003 if SIDE else 0.008) * acc)[:, None]
    s = np.clip(((RIM - o) * d).sum(1), 0, t_end)
    dist = length(o + d * s[:, None] - RIM)
    light += RIM_COL[None] * ((0.02 if SIDE else 0.05) / (1 + (dist / (0.4 if SIDE else 0.5)) ** 2))[:, None]
    return light


def render_rays(d):
    t, hit = march(CAM, d)
    p = CAM + d * t[:, None]
    mat, did = material(p)
    n = normals(p)
    col, alb = shade(p, n, mat, did)
    col += haze(CAM, d, np.minimum(t, 9))
    return dict(col=col.astype(np.float32), alb=alb.astype(np.float16), p=p.astype(np.float32),
                n=n.astype(np.float16), mat=mat.astype(np.int16))


# ================================================================ flashes
FLASH_AT = {n: (TARGET[n] if n != "kick" else head_point("kick")) for n in list(TARGET) + ["kick"]}
FLASH_OFF = {n: (DRUM[n][1] * 0.12 if n in DRUM else CYM[n][1] * 0.1 if n in CYM else V(0, 0.05, -0.15))
             for n in FLASH_AT}


def flashes(f):
    """{target: level} of the hits lit at frame f: full on the hit, a little after."""
    out = {}
    for s, limb, t in HITS:
        df = (f - s * STEP) % FRAMES
        lvl = {0: 1.0, 1: 0.3}.get(df, 0)
        if lvl:
            out[t] = max(out.get(t, 0), lvl)
    return out


def flash_light(R, f):
    p, n, alb = R["p"], R["n"].astype(np.float32), R["alb"].astype(np.float32)
    out = np.zeros_like(p)
    for t, lvl in flashes(f).items():
        L = FLASH_AT[t] + FLASH_OFF[t]
        lv = L - p
        d2 = (lv * lv).sum(1)
        ndl = np.clip((n * lv).sum(1) / np.sqrt(d2), 0, 1)
        k = 0.05 if t in CYM else 0.035
        out += alb * FLASH_COL * (lvl * k * ndl * np.exp(-d2 / 0.12) / (d2 + 0.004))[:, None]   # local: only the drum and the stick
    return out


# ================================================================ to cells
def cells(a):
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        return np.stack([cells(a[..., i]) for i in range(a.shape[2])], -1)
    return np.asarray(Image.fromarray(a, "F").resize((COLS, ROWS), Image.BOX), float)


def to_cells(rgb, black=0.3, gain=1.15, expo=0.5, lo=0.1, gamma=0.8, sat=1.1):
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


# ================================================================ code in the air
WORDS = ["if", "for", "void", "int", "def", "return", "while(1)", "{ }", "=>", "&&", "0xFF", "i++",
         "#include", "import", "HAL_Delay(1);", "git push", "fft(x)", "relu", "uint8_t", "ISR", "->",
         "::", "</>", "[i]", "!=", "<<", ">>", "/* */", "//", "make", "torch", "x += 1", "0b1010",
         "printf", "async", "self", "null", "true", "yield", "main()", "loss.backward()", "mmap", "volatile"]
SYMS = list("{}[]()<>;=+-*/&|!#%:") + ["0", "1"]
BITS = ["0", "1", "01", "10", "101", "0110"]
BURST = {"crash": 24, "snare": 7, "tom1": 6, "tom2": 6, "floor": 6, "ride": 2, "kick": 4, "hihat": 1}


def particles():
    """Every flying token over one loop, as (birth frame, x, y, vx, vy, text, life, red)."""
    rng = np.random.default_rng(7)
    out = []
    for s, limb, t in HITS:
        x0, y0 = project(FLASH_AT[t])
        for k in range(BURST[t]):
            if t == "kick":
                txt = BITS[rng.integers(len(BITS))]
                vx, vy, life = rng.uniform(-0.9, 0.9), rng.uniform(-1.0, -0.6), int(rng.integers(22, 34))
            elif t == "ride":
                txt = SYMS[rng.integers(len(SYMS))]
                vx, vy, life = rng.uniform(-1.8, -0.4), rng.uniform(-1.8, -1.1), int(rng.integers(16, 24))
            elif t == "crash":
                txt = WORDS[rng.integers(len(WORDS))] if k % 2 else SYMS[rng.integers(len(SYMS))]
                a = rng.uniform(-0.8, 0.8) * math.pi / 2
                sp = rng.uniform(1.3, 2.6)
                vx, vy, life = 2.2 * sp * math.sin(a), -sp * math.cos(a) - 0.6, int(rng.integers(24, 38))
            else:
                txt = WORDS[rng.integers(len(WORDS))] if k < 4 else SYMS[rng.integers(len(SYMS))]
                vx, vy, life = rng.uniform(-2.4, 2.4), rng.uniform(-2.5, -1.5), int(rng.integers(22, 32))
            out.append((s * STEP, x0, y0, vx, vy, txt, life, t == "crash" and k % 4 == 1))
    return out


GRAV, DRAG = 0.06, 0.97


def draw_code(ch, cl, f, parts_):
    live = []
    for b, x0, y0, vx, vy, txt, life, red in parts_:
        for loop in (0, -FRAMES):
            a = f - (b + loop)
            if 0 <= a < life:
                live.append((a, x0, y0, vx, vy, txt, life, red))
    for a, x0, y0, vx, vy, txt, life, red in sorted(live, key=lambda q: -q[0]):    # older first, newest on top
        dx = vx * (1 - DRAG ** a) / (1 - DRAG)
        dy = vy * (1 - DRAG ** a) / (1 - DRAG) + 0.5 * GRAV * a * a
        u = a / life
        cls = (("ca" if u < 0.4 else "cb") if red else
               "c1" if u < 0.22 else "c2" if u < 0.5 else "c3" if u < 0.78 else "c4")
        r = int(round(y0 + dy - 1))
        c = int(round(x0 + dx - len(txt) / 2))
        if not 0 <= r < ROWS:
            continue
        for i, x in enumerate(" " + txt + " "):          # a blank cell either side keeps it legible
            if 0 <= c - 1 + i < COLS:
                ch[r, c - 1 + i], cl[r, c - 1 + i] = (x if x != " " else NBSP), cls


CODE_PAL = {"c1": "#f2eee2", "c2": "#a9b878", "c3": "#7d8a4e", "c4": "#454d2c", "ca": "#e0303a", "cb": "#7a1e24",
            "nm": "#c8242c"}


# ================================================================ build
def pose_keys():
    return sorted(set(pose_at(f) for f in range(FRAMES)), key=str)


def build():
    CACHE.mkdir(exist_ok=True)
    tag = "_" + VIEW + ("_q" if QUICK else "")
    cache = CACHE / f"drums_render{tag}.npz"
    d = camera_rays()
    keys = pose_keys()
    print(len(keys), "poses")
    if cache.exists() and "--rerender" not in sys.argv:
        z = np.load(cache, allow_pickle=True)
        R = {k: z[k] for k in ("col", "alb", "p", "n", "mat")}
    else:
        POSE["key"] = pose_at(1)
        print("marching", len(d), "rays")
        R = render_rays(d)
        np.savez_compressed(cache, **R)
    # where the figure can be in any pose: a coarse pass per pose, joined and grown
    pcache = CACHE / f"drums_poses{tag}.npz"
    if pcache.exists() and "--rerender" not in sys.argv and "--reposes" not in sys.argv:
        z = np.load(pcache, allow_pickle=True)
        idx = z["idx"]
        PR = [{k: z[f"{k}{i}"] for k in ("col", "alb", "p", "n", "mat")} for i in range(len(keys))]
    else:
        cw, chh = W // 4, H // 4
        dc = camera_rays(cw, chh)
        who = np.zeros(cw * chh, bool)
        wid = [NAMES.index(n) for n in WHO]
        for k in keys:
            POSE["key"] = k
            t, hit = march(CAM, dc)
            m, _ = material(CAM + dc * t[:, None])
            who |= np.isin(m, wid)
        big = np.kron(who.reshape(chh, cw).astype(float), np.ones((4, 4)))
        idx = np.nonzero((blur(big, 10) > 0).ravel())[0]
        print("pose rays", len(idx))
        PR = []
        for i, k in enumerate(keys):
            POSE["key"] = k
            PR.append(render_rays(d[idx]))
            print("pose", i + 1, "/", len(keys))
        np.savez_compressed(pcache, idx=idx, **{f"{k}{i}": v for i, P in enumerate(PR) for k, v in P.items()})
    return R, idx, PR, keys


def frame_picture(R, idx, PR, keys, f):
    P = PR[keys.index(pose_at(f))]
    col = R["col"].astype(np.float64).copy()
    col[idx] = P["col"]
    img = col.reshape(H, W, 3)
    # a bright flare where the stick lands
    yy, xx = np.mgrid[:H, :W]
    for t, lvl in flashes(f).items():
        c, r = project(FLASH_AT[t])
        x, y = c / COLS * W, r / ROWS * H
        dd = (xx - x) ** 2 + ((yy - y) * 1.0) ** 2
        s = W / 800
        img += (lvl * 1.6 * np.exp(-dd / (2 * (6 * s) ** 2)))[..., None] * FLASH_COL
    return tonemap(img * EXPO)


def main():
    R, idx, PR, keys = build()
    if "--image" in sys.argv:
        for f in (0, 4, 50):
            Image.fromarray((np.clip(frame_picture(R, idx, PR, keys, f), 0, 1) * 255).astype(np.uint8)).save(
                CACHE / f"drums_img{f}.png")
        print("images written")
        return
    pics = []
    for f in range(FRAMES):
        ch, ink, T = to_cells(frame_picture(R, idx, PR, keys, f))
        if f:
            # only what really moved is redrawn: a cell keeps its first-frame glyph and ink
            # unless its tone changed a lot (an arm, a stick, a flash), which keeps the
            # per-frame layers small and the animation light for the browser
            same = (np.abs(T - pics[0][2]).max(-1) < 0.14) | (ch == pics[0][0])
            ch[same], ink[same] = pics[0][0][same], pics[0][1][same]
        pics.append((ch, ink, T))
    cent = quantise(np.concatenate([ink[ch != " "] for ch, ink, _ in pics[::8]]))
    names = [f"{'defghijlmnopqr'[i // 10]}{i % 10}" for i in range(len(cent))]
    palette = {n: "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c) for n, c in zip(names, cent)}
    palette.update(CODE_PAL)
    names = np.array(names)

    toks = particles()
    kc, kr = project(KICK_C + V(0, 0.14, -KICK_H))
    # the picture and the flying code are separate layers: the code sits in its own
    # opaque sprite on top, so the picture cells it passes over stay static
    frames, code = [], []
    for f, (ch, ink, _) in enumerate(pics):
        cl = names[((ink[..., None, :] - cent[None, None]) ** 2).sum(-1).argmin(-1)].astype("<U2")
        ch = ch.copy()
        cl[ch == " "] = ""
        g = Grid(COLS, ROWS)
        g.paste(ch, cl)
        frames.append(g)
        g = Grid(COLS, ROWS)
        draw_code(g.ch, g.cl, f, toks)
        code.append(g)
    name = Grid(COLS, ROWS)
    word = "ARDA ÇABUK"                      # the name on the kick's front head, over any flying bits
    nc, nr = int(round(kc - len(word) / 2)), int(round(kr))
    name.text(nc, nr, word, "nm")
    name.cover(nc - 1, nr, nc + len(word), nr)          # the gaps hide the picture too
    static, sprites = pack(frames)
    print(f"{len(sprites)} sprites")
    movie = Movie(COLS, ROWS, pad=(12, 14), palette=palette, bold=True, font_px=17,
                  title="Arda Çabuk — embedded software and machine learning",
                  desc="A ray-marched practice room drawn in ASCII. Someone with short curly dark-blond hair, a pale hoodie "
                       "and red headphones plays a red drum kit under a warm spotlight, with a red light glowing behind; "
                       "the front head of the bass drum reads ARDA ÇABUK. Every hit flashes the drum it lands on and "
                       "throws code into the air: keywords and symbols fly up out of the snare and the toms, the crash "
                       "throws a fountain of them, the ride flicks a single symbol and the kick lets bits drift up, all "
                       "arcing and fading from white to olive. Three bars of groove end in a fill around the toms. A black "
                       "cat sits on the amp beside the kit.")
    movie.add(static)
    for fr, hold in sprites:
        movie.add(fr, FPS, hold=hold)
    movie.add(code, FPS, opaque=True)
    movie.add(name, opaque=True)
    if "--preview" in sys.argv:
        movie.preview("drums.png", times=(0, 0.5, 6.25))
    movie.write(f"hero_drums_{VIEW}.svg")


if __name__ == "__main__":
    main()
