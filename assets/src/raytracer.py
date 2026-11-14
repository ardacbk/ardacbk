"""RayTracing card: a small path tracer converging.

A real (tiny) path tracer in numpy: a matte and a glossy sphere on a ground
plane, lit by two glowing spheres (one in view, one above the camera). Each pass shoots one jittered ray per pixel and
follows it for a few bounces; the picture shown is the running average of all
passes so far, so it starts as speckle and settles. Then the camera moves and
the accumulation starts over, which is also where the loop restarts.

The image has 4 x 8 pixels per character cell. A cell's brightness picks a
character from a density ramp and a tone from the palette.
"""
import sys

import numpy as np

from engine import CARD, SX, SY, Grid, Movie, cell_mean

COLS, ROWS = CARD
PIC_ROWS = ROWS - 2                       # picture area; the last rows hold the readout
W, H = COLS * SX, PIC_ROWS * SY
FPS = 8
STAGES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]    # samples shown, one stage per tick
HOLD_LAST = 14                            # ticks the converged picture stays up
RAMP = " .:-=+*#%@"
BOUNCES = 4
EXPOSURE, BLACK = 1.5, 0.2
GLOSS = 0.4                               # share of rays a glossy surface reflects specularly

# name: centre, radius, albedo, roughness of the specular part (None = purely diffuse), emission
SPHERES = [
    ("matte", np.array([-1.25, 1.0, -4.6]), 1.0, 0.72, None, 0.0),
    ("glossy", np.array([1.15, 0.75, -3.6]), 0.75, 0.85, 0.05, 0.0),
    ("glow", np.array([-0.15, 0.3, -2.9]), 0.3, 0.0, None, 9.0),       # the emissive sphere in view
    ("key", np.array([-3.0, 6.5, -1.0]), 1.8, 0.0, None, 8.5),         # a second, larger one above the camera
]
GROUND_ALBEDO = 0.3


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def camera_rays(eye, target, rng):
    """One jittered ray per pixel."""
    fwd = normalize(target - eye)
    right = normalize(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
    up = np.cross(right, fwd)
    half_h = np.tan(np.radians(17.5))
    half_w = half_h * W / H
    px = (np.arange(W)[None, :] + rng.random((H, W))) / W * 2 - 1
    py = 1 - (np.arange(H)[:, None] + rng.random((H, W))) / H * 2
    d = fwd + px[..., None] * half_w * right + py[..., None] * half_h * up
    return np.broadcast_to(eye, (H * W, 3)).copy(), normalize(d.reshape(-1, 3))


def intersect(ro, rd):
    """Nearest hit: distance, object index (-1 none, len(SPHERES) = ground), normal."""
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
    hit = (rd[:, 1] < 0) & (t > 1e-3) & (t < best)
    best[hit], idx[hit] = t[hit], len(SPHERES)
    normal[hit] = (0.0, 1.0, 0.0)
    return best, idx, normal


def random_unit(n, rng):
    v = rng.normal(size=(n, 3))
    return normalize(v)


def one_pass(eye, target, rng):
    """One sample per pixel -> (radiance image, mask of pixels that see the light directly)."""
    ro, rd = camera_rays(eye, target, rng)
    radiance = np.zeros(H * W)
    throughput = np.ones(H * W)
    alive = np.arange(H * W)
    sees_light = np.zeros(H * W, bool)
    for bounce in range(BOUNCES):
        t, idx, normal = intersect(ro, rd)
        miss = idx < 0
        sky = 0.012 + 0.03 * np.clip(rd[:, 1], 0, 1)
        radiance[alive[miss]] += throughput[alive[miss]] * sky[miss]
        albedo = np.full(len(ro), GROUND_ALBEDO)
        rough = np.full(len(ro), np.nan)
        emit = np.zeros(len(ro))
        for i, (_, _, _, a, rg, e) in enumerate(SPHERES):
            m = idx == i
            albedo[m], emit[m] = a, e
            rough[m] = np.nan if rg is None else rg
        radiance[alive] += throughput[alive] * emit
        if bounce == 0:
            sees_light[alive[emit > 0]] = True
        go = ~miss & (emit == 0)
        pos = ro[go] + rd[go] * t[go, None]
        n = normal[go]
        # a glossy surface reflects like a mirror for some rays and scatters the rest
        mirror = ~np.isnan(rough[go]) & (rng.random(len(pos)) < GLOSS)
        jitter = random_unit(len(pos), rng)
        diffuse_dir = normalize(n + jitter)
        refl = rd[go] - 2 * (rd[go] * n).sum(1, keepdims=True) * n
        refl = normalize(refl + np.nan_to_num(rough[go])[:, None] * jitter)
        new_dir = np.where(mirror[:, None], refl, diffuse_dir)
        alive = alive[go]
        throughput[alive] *= albedo[go]
        ro, rd = pos + n * 1e-3, new_dir
    return radiance.reshape(H, W), sees_light.reshape(H, W)


def tone_map(img):
    """Exposure, gamma, then a black point so unlit areas stay empty."""
    v = (1 - np.exp(-EXPOSURE * img)) ** (1 / 2.2)
    return np.clip((v - BLACK) / (1 - BLACK), 0, 1)


def to_grid(img, light_mask):
    lum = cell_mean(tone_map(img))
    level = np.clip((lum * len(RAMP)).astype(int), 0, len(RAMP) - 1)
    ch = np.array(list(RAMP))[level]
    cl = np.full(lum.shape, "t1", dtype="<U2")
    for edge, name in ((0.22, "t2"), (0.4, "t3"), (0.6, "t4"), (0.8, "t5")):
        cl[lum >= edge] = name
    glow = cell_mean(light_mask.astype(float))
    cl[glow > 0.25] = "b"
    cl[glow > 0.6] = "a"
    cl[ch == " "] = ""
    g = Grid(COLS, ROWS)
    g.paste(ch, cl)
    return g


def cycle(eye, target, seed):
    """Frames of one accumulation run: the average after each sample count in STAGES."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W))
    light = np.zeros((H, W))
    frames, done = [], 0
    for n in STAGES:
        while done < n:
            img, mask = one_pass(eye, target, rng)
            acc += img
            light += mask
            done += 1
        g = to_grid(acc / done, light / done)
        g.text(2, ROWS - 1, "accumulation on", "t2")
        g.text(21, ROWS - 1, f"samples {done:04d}", "t3")
        if n == STAGES[0]:
            g.rtext(COLS - 3, ROWS - 1, "camera moved -> reset", "t3")
        frames.append(g)
    return frames


def build():
    target = np.array([-0.15, 0.68, -4.2])
    frames = cycle(np.array([-0.4, 1.45, 0.9]), target, 1) + cycle(np.array([1.7, 1.3, 0.6]), target, 2)
    hold = ([1] * (len(STAGES) - 1) + [HOLD_LAST]) * 2
    hold[0] = hold[len(STAGES)] = 3          # let the first speckle and the reset note register
    m = Movie(COLS, ROWS, title="A path tracer converging",
              desc="Two spheres on a ground plane, lit by a glowing sphere, rendered as characters. The picture "
                   "starts as speckle and settles as samples accumulate; then the camera moves and it starts over.")
    m.add(frames, FPS, hold=hold)
    return m


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("raytracer.png", times=(0, 0.6, 1.0, 1.6, 3.0, 4.5))
        m.preview("raytracer-light.png", times=(3.0,), theme="light")
    m.write("raytracer.svg")
