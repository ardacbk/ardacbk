"""CKD card: reading a clinical table without filling the holes.

A small missing-aware logistic model is fitted on synthetic patients drawn from
two clusters (ckd-like, healthy-like) with random missingness. It never imputes:
a missing feature is masked out and the remaining weighted evidence is
renormalised by (features / present features). The animation reads the rows one
by one: present values light up left to right, missing ones turn into a green
mask mark, then the row's probability appears with a bar.

Under the table the model's evidence for the row being read is drawn as bars,
one per feature: up (red) pushes towards kidney disease, down towards healthy,
and a masked feature contributes nothing. That is what "masked, not imputed"
means: the hole adds no evidence, and the rest is rescaled.
"""
import sys

import numpy as np

from engine import CARD_FINE, FINE, Grid, Movie

COLS, ROWS = CARD_FINE
FPS = 10
FEATS = ["age", "bp", "sg", "al", "bgr", "bu", "sc", "hemo"]
# (mean, sd) per cluster: ckd-like, healthy-like, in real units
CKD = [(58, 13), (80, 14), (1.012, .004), (2.4, 1.2), (160, 65), (75, 38), (4.2, 3.0), (10.4, 2.1)]
OK = [(46, 14), (72, 8), (1.021, .003), (0.1, .25), (108, 18), (30, 9), (0.9, .22), (14.6, 1.4)]
FMT = [lambda v: f"{v:.0f}", lambda v: f"{v:.0f}", lambda v: f"{v:.3f}", lambda v: f"{max(v, 0):.0f}",
       lambda v: f"{v:.0f}", lambda v: f"{v:.0f}", lambda v: f"{max(v, .4):.1f}", lambda v: f"{v:.1f}"]
WID = [5, 5, 6, 4, 6, 6, 6, 6, 7]          # last one is p(ckd)
GAP = 3
C0 = 4
PBAR = 12                                  # width of the probability bar
AXIS, BAR_H = 18, 5                        # row of the evidence axis, bar height in rows each way


def draw(rng, cluster, n):
    p = CKD if cluster else OK
    x = np.stack([rng.normal(m, s, n) for m, s in p], 1)
    x[:, 3] = np.clip(np.round(x[:, 3]), 0, 5)
    return x


def renorm_score(w, b, z, present):
    """Evidence from the present features only, rescaled to a full row; no imputation."""
    k = present.sum(1).clip(1)
    return (np.where(present, z * w, 0).sum(1)) * (len(w) / k) + b


def fit(rng):
    n = 600
    y = (rng.random(n) < .5).astype(float)
    x = np.where(y[:, None] > 0, draw(rng, 1, n), draw(rng, 0, n))
    present = rng.random(x.shape) > .27
    mu = np.nanmean(np.where(present, x, np.nan), 0)
    sd = np.nanstd(np.where(present, x, np.nan), 0)
    z = (x - mu) / sd
    w, b = np.zeros(len(FEATS)), 0.0
    for _ in range(600):
        p = 1 / (1 + np.exp(-renorm_score(w, b, z, present)))
        g = (p - y)[:, None]
        k = (len(w) / present.sum(1).clip(1))[:, None]
        w -= .2 * (np.where(present, z, 0) * k * g).mean(0) + .05 * w
        b -= .2 * g.mean()
    p = 1 / (1 + np.exp(-renorm_score(w, b, z, present)))
    print(f"train accuracy with masks: {((p > .5) == (y > .5)).mean():.3f}")
    return w, b, mu, sd


def patients(rng, w, b, mu, sd):
    labels = [1, 0, 1, 1, 0, 1, 0, 0, 1]
    rows = []
    for lab in labels:
        x = draw(rng, lab, 1)[0]
        while True:
            present = rng.random(len(FEATS)) > .27
            if 4 <= present.sum() <= 7:
                break
        z = (x - mu) / sd
        s = renorm_score(w, b, z[None], present[None])[0]
        ev = np.where(present, z * w, 0) * (len(w) / present.sum())   # each feature's share of the evidence
        rows.append((x, present, 1 / (1 + np.exp(-s)), ev))
    return rows


def build():
    rng = np.random.default_rng(11)
    w, b, mu, sd = fit(rng)
    rows = patients(rng, w, b, mu, sd)
    ends = np.cumsum([w_ + GAP for w_ in WID]) - GAP          # right edge offset of each column
    right = [C0 + int(e) - 1 for e in ends]                    # absolute right column per field
    r_head, r_rule, r0, r_leg = 1, 2, 3, ROWS - 1
    pb0 = right[8] + 3
    centre = [right[j] - WID[j] // 2 for j in range(8)]
    ev_max = max(np.abs(e).max() for *_, e in rows)
    miss_total = sum((~p).sum() for _, p, _, _ in rows)
    print(f"missing cells: {miss_total}/{len(rows) * 8}")
    for x, p, pr, _ in rows:
        print([FMT[j](x[j]) if p[j] else "?" for j in range(8)], f"{pr:.2f}")

    def base():
        g = Grid(COLS, ROWS)
        for j, name in enumerate(FEATS + ["p(ckd)"]):
            g.rtext(right[j], r_head, name, "t2")
        for c in range(C0 - 1, right[-1] + 2):
            g.put(c, r_rule, "-", "t1")
        g.text(C0 - 1, r_leg, "? missing -> masked in attention, not imputed", "t2")
        for c in range(C0 - 1, right[7] + 2):
            g.put(c, AXIS, "-", "t1")
        g.text(C0 - 1, AXIS - BAR_H, "ckd", "t2")
        g.text(C0 - 1, AXIS + BAR_H, "ok", "t2")
        g.text(right[7] + 4, AXIS, "evidence", "t2")
        for i, (x, p, pr, _) in enumerate(rows):
            for j in range(8):
                if p[j]:
                    g.rtext(right[j], r0 + i, FMT[j](x[j]), "t3")
                else:
                    g.rtext(right[j], r0 + i, "?", "t2")
        return g

    def pbar(g, r, pr, cls):
        k = int(round(pr * PBAR))
        g.text(pb0, r, "#" * k + "." * (PBAR - k), cls)

    def mark(g, i, masks=True, pred=None):
        x, p, pr, _ = rows[i]
        if masks:
            for j in range(8):
                if not p[j]:
                    g.rtext(right[j], r0 + i, "--", "k")
        if pred:
            g.rtext(right[8], r0 + i, f"{pr:.2f}", "b" if pr > .5 else "t3")
            pbar(g, r0 + i, pr, "b" if pr > .5 else "t2")

    def evidence(g, i, upto):
        """The row's evidence bars, for the features read so far."""
        x, p, pr, ev = rows[i]
        for j in range(min(upto + 1, 8)):
            c = centre[j]
            if not p[j]:
                g.ctext(c, AXIS, "mask", "k")
                continue
            h = ev[j] / ev_max * BAR_H
            n = int(round(abs(h)))
            up = h > 0
            for k in range(n):
                r = AXIS - 1 - k if up else AXIS + 1 + k
                g.text(c - 1, r, "###", "a" if up else "t4")
            if n == 0:
                g.text(c - 1, AXIS - 1 if up else AXIS + 1, "___" if up else "---", "a" if up else "t4")

    # timeline (seconds): 0-1 rest, row i read during [1+i, 2+i], hold to 11.5, clear, loop = 12 s
    # layer 1: finished rows (masks + dim predictions), 2 steps per second
    done = []
    for k in range(24):
        g = Grid(COLS, ROWS)
        n_done = 0 if (k >= 23 or k < 4) else min((k - 4) // 2 + 1, len(rows))
        for i in range(n_done):
            mark(g, i, True, "b")
        done.append(g)
    # layer 2: the row being read, 10 fps
    cur = []
    for f in range(120):
        g = Grid(COLS, ROWS)
        i, s = divmod(f - 10, 10)
        if 0 <= i < len(rows):
            x, p, pr, _ = rows[i]
            r = r0 + i
            for j in range(8):
                if p[j]:
                    g.rtext(right[j], r, FMT[j](x[j]), "t5" if s >= j else "t3")
                else:
                    g.rtext(right[j], r, "--" if s >= j else "?", "k" if s >= j else "t2")
            if s >= 8:
                g.rtext(right[8], r, f"{pr:.2f}", "a" if pr > .5 else "t4")
                pbar(g, r, pr, "a" if pr > .5 else "t4")
            g.put(1, r, ">", "t4")
            evidence(g, i, s)
        cur.append(g)
    frames = [done, cur]

    m = Movie(COLS, ROWS, cell=FINE, title="NAIM on chronic kidney disease",
              desc="A table of illustrative patients with blood and urine features, about a quarter of the values "
                   "missing. A cursor reads the rows one by one: present values light up, each missing value is "
                   "masked in green rather than imputed, and a probability of kidney disease appears with a bar, in "
                   "red when it is above one half. Under the table, bars show each feature's evidence for the row: "
                   "up towards disease, down towards healthy, nothing for a masked feature.")
    m.add(base())
    m.add(done, 2, opaque=True)
    m.add(cur, FPS, opaque=True)
    return m, len(cur)


if __name__ == "__main__":
    m, n = build()
    print(f"{n} frames, {n / FPS:.1f}s")
    if "--preview" in sys.argv:
        m.preview("ckd.png", times=(0.3, 1.8, 3.9, 6.2, 8.4, 9.9, 10.8, 11.9))
        m.preview("ckd-light.png", times=(2.5, 9.8), theme="light")
    m.write("ckd.svg")
