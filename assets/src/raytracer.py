"""RayTracing card: a small path tracer converging.

A real (tiny) path tracer in numpy, after the project it illustrates: a
glossy sphere, a matte one and a glowing red one hovering above a dark
mirror floor.
The scene is kept to a few large shapes on purpose: at 68 x 16 characters
fine detail turns to noise, while a shaded sphere, the whole-shape copy of it
in the floor and a red glow still read. The red sphere's light is carried as
a separate channel through every bounce, so the red it casts on the others
and in the floor is printed in the accent tone.

Each pass shoots one jittered ray per pixel and follows it for a few bounces;
the picture shown is the running average of all passes, so it starts as
speckle and settles. When the camera moves the average is thrown away: the
frames in motion are single noisy passes, and accumulation starts again once
the camera stops. The loop orbits between two viewpoints and back.

The image has 4 x 8 pixels per character cell. Characters come from the
glyph matcher, so a cell gets the glyph whose shape is closest to the pixels
under it (silhouettes print as / \\ ( ) _), and a tone from its brightness.
"""
import math
import sys

import numpy as np

from engine import CARD_FINE, FINE, SX, SY, Grid, Movie, Printer, cell_mean

COLS, ROWS = CARD_FINE                    # fine cells: about twice the characters of a standard card
PIC_ROWS = ROWS - 2                       # picture area; the last rows hold the readout
W, H = COLS * SX, PIC_ROWS * SY
FPS = 8
STAGES = [1, 2, 4, 8, 16, 32, 64, 128, 256]   # samples shown, one stage per tick
HOLD_LAST = 16                            # ticks the converged picture stays up
MOVE = 4                                  # frames the camera spends moving (one noisy pass each)
BOUNCES = 5
EXPOSURE, BLACK = 0.9, 0.12

# name: centre, radius, albedo, kind, roughness, emission (white, red)
SPHERES = [
    ("ivory", np.array([0.2, 1.0, -5.0]), 1.0, 0.8, "glossy", 0.0, (0, 0)),
    ("slate", np.array([-2.05, 0.8, -5.6]), 0.8, 0.55, "diffuse", 0, (0, 0)),
    ("ember", np.array([2.2, 0.78, -5.3]), 0.4, 0.0, "diffuse", 0, (0, 2.0)),   # hovers, so its reflection stands apart
    ("key", np.array([-6.0, 6.0, -1.5]), 2.5, 0.0, "diffuse", 0, (9.0, 0)),    # area light up and to the left
]
FLOOR = (0.12, 0.12)                      # checker albedos (equal: a plain floor reads better in print)
TILE = 1.6                                # checker size
CLAMP = 3.0                               # most radiance one sample may carry
GLOSS = 0.35                              # share of rays the glossy sphere's clear coat reflects
FLOOR_GLOSS, FLOOR_ROUGH = 0.7, 0.03      # share of floor rays reflected, and how sharply
LIGHTS = [i for i, s_ in enumerate(SPHERES) if sum(s_[6]) > 0]
TARGET = np.array([0.0, 0.55, -5.0])
VIEWS = [(-0.25, 0.6, 4.7), (0.3, 0.75, 4.5)]   # (azimuth, height, distance) around TARGET

PRINT = Printer()


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def eye_at(az, height, dist):
    return TARGET + np.array([math.sin(az) * dist, height - TARGET[1], math.cos(az) * dist])


def camera_rays(eye, rng):
    """One jittered ray per pixel."""
    fwd = normalize(TARGET - eye)
    right = normalize(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
    up = np.cross(right, fwd)
    half_h = np.tan(np.radians(19))
    half_w = half_h * (W * 8.4 / SX) / (H * 17.0 / SY)           # cell aspect is the same at any scale
    px = (np.arange(W)[None, :] + rng.random((H, W))) / W * 2 - 1
    py = 1 - (np.arange(H)[:, None] + rng.random((H, W))) / H * 2
    d = fwd + px[..., None] * half_w * right + py[..., None] * half_h * up
    return np.broadcast_to(eye, (H * W, 3)).copy(), normalize(d.reshape(-1, 3))


def intersect(ro, rd):
    """Nearest hit: distance, object index (-1 none, len(SPHERES) = floor), normal."""
    n = len(ro)
    best = np.full(n, np.inf)
    idx = np.full(n, -1)
    normal = np.zeros((n, 3))
    for i, (_, c, r, *_rest) in enumerate(SPHERES):
        oc = ro - c
        b = (oc * rd).sum(1)
        disc = b * b - ((oc * oc).sum(1) - r * r)
        t = -b - np.sqrt(np.maximum(disc, 0))
        hit = (disc > 0) & (t > 1e-3) & (t < best)
        best[hit], idx[hit] = t[hit], i
        normal[hit] = (ro[hit] + rd[hit] * t[hit, None] - c) / r
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -ro[:, 1] / rd[:, 1]
    hit = (rd[:, 1] < 0) & (t > 1e-3) & (t < best) & (t < 40)
    best[hit], idx[hit] = t[hit], len(SPHERES)
    normal[hit] = (0.0, 1.0, 0.0)
    return best, idx, normal


def one_pass(eye, rng):
    """One sample per pixel -> radiance (H, W, 2): white light and the ember's red light."""
    ro, rd = camera_rays(eye, rng)
    radiance = np.zeros((H * W, 2))
    throughput = np.ones(H * W)
    alive = np.arange(H * W)
    spec = np.ones(H * W, bool)          # camera rays and mirror bounces see emitters directly
    for _ in range(BOUNCES):
        t, idx, normal = intersect(ro, rd)
        miss = idx < 0
        up = np.clip(rd[:, 1], 0, 1)
        sky = 0.004 + 0.05 * np.exp(-up / 0.04)                        # night, a faint glow at the horizon
        radiance[alive[miss], 0] += throughput[alive[miss]] * sky[miss]
        albedo = np.zeros(len(ro))
        metal = np.zeros(len(ro), bool)
        rough = np.zeros(len(ro))
        emit = np.zeros((len(ro), 2))
        for i, (_, _, _, a, kind, rg, e) in enumerate(SPHERES):
            m = idx == i
            albedo[m], metal[m], rough[m], emit[m] = a, kind == "metal", rg, e
            if kind == "glossy":                # a clear coat: a share of rays reflect, the rest scatter
                coat = m & (rng.random(len(ro)) < GLOSS)
                metal[coat], albedo[coat] = True, 0.9
        fl = idx == len(SPHERES)
        if fl.any():
            p = ro[fl] + rd[fl] * t[fl, None]
            check = (np.floor(p[:, 0] / TILE) + np.floor(p[:, 2] / TILE)) % 2
            albedo[fl] = np.where(check > 0, FLOOR[1], FLOOR[0])
            gloss = rng.random(fl.sum()) < FLOOR_GLOSS
            metal[np.nonzero(fl)[0][gloss]] = True
            rough[fl] = FLOOR_ROUGH
            albedo[np.nonzero(fl)[0][gloss]] = 0.8
        radiance[alive] += throughput[alive, None] * emit * spec[:, None]
        go = ~miss & (emit.sum(1) == 0)
        pos = ro[go] + rd[go] * t[go, None]
        n = normal[go]
        # next-event estimation: diffuse hits sample each light directly with a shadow ray
        dif = ~metal[go]
        if dif.any():
            q, nq = pos[dif] + n[dif] * 1e-3, n[dif]
            for li in LIGHTS:
                _, c, r, _, _, _, e = SPHERES[li]
                to = c - q
                d2 = (to * to).sum(1)
                cmax = np.sqrt(np.clip(1 - r * r / d2, 0, 1))
                u1, u2 = rng.random(len(q)), rng.random(len(q))
                ct = 1 - u1 * (1 - cmax)                                  # uniform direction in the light's cone
                st, ph = np.sqrt(1 - ct * ct), 2 * np.pi * u2
                w = normalize(to)
                a_ = np.where(np.abs(w[:, :1]) > 0.9, [[0, 1, 0]], [[1, 0, 0]])
                uu = normalize(np.cross(a_, w))
                vv = np.cross(w, uu)
                ld = uu * (st * np.cos(ph))[:, None] + vv * (st * np.sin(ph))[:, None] + w * ct[:, None]
                cos = np.clip((ld * nq).sum(1), 0, None)
                _, hid, _ = intersect(q, ld)
                vis = (hid == li) & (cos > 0)
                k = albedo[go][dif] / np.pi * cos * 2 * np.pi * (1 - cmax) * vis
                radiance[alive[go][dif]] += (throughput[alive[go][dif]] * k)[:, None] * np.array(e)[None]
        jitter = normalize(rng.normal(size=(len(pos), 3)))
        diffuse_dir = normalize(n + jitter)
        refl = rd[go] - 2 * (rd[go] * n).sum(1, keepdims=True) * n
        refl = normalize(refl + rough[go][:, None] * jitter)
        new_dir = np.where(metal[go][:, None], refl, diffuse_dir)
        spec = metal[go]
        alive = alive[go]
        throughput[alive] *= albedo[go]
        ro, rd = pos + n * 1e-3, new_dir
    tot = radiance.sum(1, keepdims=True)
    radiance *= np.minimum(1, CLAMP / np.maximum(tot, 1e-9))           # tame fireflies from rare bright paths
    return radiance.reshape(H, W, 2)


def tone_map(img):
    """Exposure, gamma, then a black point so unlit areas stay empty."""
    v = (1 - np.exp(-EXPOSURE * img)) ** (1 / 2.2)
    return np.clip((v - BLACK) / (1 - BLACK), 0, 1)


def to_grid(img):
    """Average radiance (H, W, 2) -> characters by shape, tones by brightness, red where the ember's light dominates."""
    tot = img.sum(-1)
    v = tone_map(tot)
    ch, lum = PRINT(v)
    red = cell_mean(img[..., 1]) / np.maximum(cell_mean(tot), 1e-6)
    cl = np.full(lum.shape, "t1", dtype="<U2")
    for edge, name in ((0.16, "t2"), (0.32, "t3"), (0.5, "t4"), (0.7, "t5")):
        cl[lum >= edge] = name
    cl[(red > 0.45) & (lum > 0.08)] = "b"
    cl[(red > 0.45) & (lum > 0.6)] = "a"            # the glowing sphere itself; its reflection stays dim red
    cl[ch == " "] = ""
    g = Grid(COLS, ROWS)
    g.paste(ch, cl)
    return g


def readout(g, left, right, cls="t3"):
    g.text(2, ROWS - 1, left, "t2")
    g.text(21, ROWS - 1, right, cls)


def converge(eye, seed):
    """Frames of one accumulation run: the average after each sample count in STAGES."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W, 2))
    frames, done = [], 0
    for n in STAGES:
        while done < n:
            acc += one_pass(eye, rng)
            done += 1
        g = to_grid(acc / done)
        readout(g, "accumulation on", f"samples {done:04d}")
        frames.append(g)
    return frames


def move(a, b, seed):
    """The camera gliding from view a to view b: one noisy pass per frame, nothing accumulated."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(1, MOVE + 1):
        s = k / (MOVE + 1)
        s = s * s * (3 - 2 * s)
        g = to_grid(one_pass(eye_at(*[x + (y - x) * s for x, y in zip(a, b)]), rng))
        readout(g, "accumulation on", "camera moving -> reset", "a")
        frames.append(g)
    return frames


def build():
    a, b = VIEWS
    frames = converge(eye_at(*a), 1) + move(a, b, 2) + converge(eye_at(*b), 3) + move(b, a, 4)
    one = [2] + [1] * (len(STAGES) - 2) + [HOLD_LAST]
    hold = one + [1] * MOVE + one + [1] * MOVE
    m = Movie(COLS, ROWS, cell=FINE, title="A path tracer converging",
              desc="A glossy sphere, a matte one and a glowing red one hovering above a dark mirror floor, path traced and "
                   "drawn as characters. The picture starts as speckle and settles into shaded spheres with their "
                   "reflections below them as samples accumulate, the red glow tinting its neighbour. Then the camera glides to a new view, the picture turns to noise while it moves, and "
                   "accumulation starts over.")
    m.add(frames, FPS, hold=hold)
    return m


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("raytracer.png", times=(0, 0.6, 1.0, 1.3, 3.4, 3.7, 6.0))
        m.preview("raytracer-light.png", times=(3.0,), theme="light")
    m.write("raytracer.svg")
