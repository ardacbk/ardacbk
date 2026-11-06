"""Six animated project cards. usage: python cards.py [name ...]  (default: all)"""
import math
import random
import sys

from kit import (ACCENT, BG, DIM, EASE, EDGE, FAINT, INK, MUTED, SAND, anim, anim_tr,
                 card, f, fades, windows, write)


# ---------------------------------------------------------------------------
# 01 · TURBINES: scrolling spectrogram + LSTM-AE reconstruction error
# ---------------------------------------------------------------------------
def turbines():
    random.seed(28)
    T = 10
    CW, RH, NR = 8, 7, 16
    X0, X1 = 24, 576
    VIS = (X1 - X0) // CW  # visible columns
    L = VIS  # loop length in columns
    SPEC_BOTTOM = 166
    A0, AW = 30, 11  # anomaly window (columns)

    transients = {c for c in range(L) if random.random() < 0.05}

    def mag(c, r):
        cyc = 2 * math.pi * c / L
        f0 = 1.6 + 0.25 * math.sin(cyc * 3)
        m = 0.02 + 0.12 * random.random() ** 2
        for k, amp in ((1, 1.0), (2, 0.7), (3, 0.42), (4, 0.25), (5, 0.12)):
            mod = 0.6 + 0.4 * math.sin(cyc * 2 + k * 1.3)
            m += amp * mod * math.exp(-((r - f0 * k) ** 2) / (2 * 0.42 ** 2))
        if c in transients:
            m += 0.35 * math.exp(-r / 9) * random.random()
        an = 0.0
        if A0 <= c < A0 + AW:
            w = math.sin(math.pi * (c - A0 + 0.5) / AW)
            an = w * (0.95 * math.exp(-((r - 12.4) ** 2) / (2 * 0.8 ** 2)) + 0.16 * random.random())
        return min(1, m), min(1, an)

    cells = [[mag(c, r) for r in range(NR)] for c in range(L)]
    err = []
    for c in range(L):
        e = 0.13 + 0.07 * random.random()
        d = (c - (A0 + AW / 2 - 0.5)) / 2.6
        e += 0.72 * math.exp(-d * d / 2)
        err.append(e)

    rects = []
    for copy in range(2):
        for c in range(L):
            x = X0 + (copy * L + c) * CW
            for r in range(NR):
                m, an = cells[c][r]
                y = SPEC_BOTTOM - (r + 1) * RH
                if an > 0.22 and an > m * 0.6:
                    rects.append(f'<rect x="{x}" y="{y}" width="7" height="6" fill="{ACCENT}" opacity="{f(0.35 + 0.65 * an, 2)}"/>')
                elif m > 0.09:
                    rects.append(f'<rect x="{x}" y="{y}" width="7" height="6" opacity="{f(min(0.95, m ** 1.1), 2)}"/>')

    EY0, EY1 = 196, 252

    def ey(e):
        return EY1 - e * (EY1 - EY0) / 1.0

    pts = []
    hot = []
    for copy in range(2):
        seg = []
        for c in range(L):
            x = X0 + (copy * L + c) * CW + 3.5
            y = ey(err[c])
            pts.append(f"{'M' if not pts else 'L'}{f(x)} {f(y)}")
            if err[c] > 0.5:
                seg.append(f"{'M' if not seg else 'L'}{f(x)} {f(y)}")
        hot.append(" ".join(seg))
    thr_y = ey(0.5)

    P = 50  # playhead column (visible index)
    px = X0 + P * CW + 3.5
    center = A0 + AW / 2 - 0.5
    tc = ((center - P) % L) / L
    half = 3.2 / L
    flag = fades([(tc - half, tc + half)], ramp=0.012)

    body = f"""
<clipPath id="view"><rect x="{X0}" y="50" width="{X1 - X0}" height="214"/></clipPath>
<g clip-path="url(#view)">
  <g>
    {anim_tr("translate", [(0, "0 0"), (1, f"{-L * CW} 0")], T)}
    <g fill="{INK}">{"".join(rects)}</g>
    <path d="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="{" ".join(hot)}" fill="none" stroke="{ACCENT}" stroke-width="2.2" stroke-linejoin="round"/>
  </g>
</g>
<line x1="{X0}" y1="{f(thr_y)}" x2="{X1}" y2="{f(thr_y)}" stroke="{DIM}" stroke-dasharray="3 4"/>
<text x="{X1}" y="{f(thr_y - 5)}" font-size="11.5" fill="{DIM}" text-anchor="end">threshold</text>
<text x="{X0}" y="184" font-size="12" fill="{MUTED}">spectrogram  ·  28 channels → 426-d FFT features</text>
<text x="{X0}" y="{EY0 + 8}" font-size="12" fill="{MUTED}">LSTM-AE reconstruction error</text>
<line x1="{f(px)}" y1="52" x2="{f(px)}" y2="{EY1 + 4}" stroke="{DIM}" stroke-width="1">
  {anim("stroke", [(v[0], ACCENT if v[1] else DIM) for v in windows([(tc - half, tc + half)])], T, discrete=True)}
</line>
<g opacity="0">
  {anim("opacity", flag, T)}
  <rect x="{f(px + 8)}" y="56" width="74" height="18" rx="9" fill="{ACCENT}"/>
  <text x="{f(px + 45)}" y="69" font-size="11.5" fill="{BG}" text-anchor="middle" font-weight="700" letter-spacing="1.5">ANOMALY</text>
</g>
"""
    write("card-turbines.svg", card(
        "03", "TURBINES", "graduation project",
        "Turbine condition monitoring on Jetson",
        "FFT features · spectrogram CNN · LSTM autoencoder",
        body))


# ---------------------------------------------------------------------------
# 02 · ALIGNMENT: an RL agent rotates a 3D plane onto its target
# ---------------------------------------------------------------------------
def alignment():
    random.seed(2025)
    T = 9
    CX, CY, S = 215, 168, 110
    ELEV = math.radians(30)  # camera looks down 30° from horizontal
    D, F = 6.5, 6.5

    def rot(a, b, g):
        ca, sa, cb, sb, cg, sg = (math.cos(a), math.sin(a), math.cos(b), math.sin(b),
                                  math.cos(g), math.sin(g))
        rx = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]
        ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
        rz = [[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]]
        mul = lambda A, B: [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return mul(rz, mul(ry, rx))

    def proj(p):
        x, y, z = p
        up = y * math.sin(ELEV) + z * math.cos(ELEV)
        depth = y * math.cos(ELEV) - z * math.sin(ELEV)
        s = F / (D + depth)
        return CX + x * s * S, CY - up * s * S

    def apply(R, p):
        return [sum(R[i][k] * p[k] for k in range(3)) for i in range(3)]

    ticks = [-1 + 2 * i / 6 for i in range(7)]

    def plane_paths(R):
        grid = []
        for u in ticks[1:-1]:
            for a, b in (((u, -1, 0), (u, 1, 0)), ((-1, u, 0), (1, u, 0))):
                (x1, y1), (x2, y2) = proj(apply(R, a)), proj(apply(R, b))
                grid.append(f"M{f(x1)} {f(y1)}L{f(x2)} {f(y2)}")
        corners = [proj(apply(R, c)) for c in ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))]
        border = "M" + "L".join(f"{f(x)} {f(y)}" for x, y in corners) + "Z"
        (nx0, ny0), (nx1, ny1) = proj(apply(R, (0, 0, 0))), proj(apply(R, (0, 0, 0.85)))
        normal = f"M{f(nx0)} {f(ny0)}L{f(nx1)} {f(ny1)}"
        return " ".join(grid), border, normal

    # target (identity orientation)
    _, t_border, t_normal = plane_paths(rot(0, 0, 0))

    # agent episode: damped corrections with decaying jitter
    th = [math.radians(-20), math.radians(24), math.radians(32)]
    vel = [0, 0, 0]
    w, z, dt = 0.8, 0.3, 0.17
    states = [tuple(th)]
    for n in range(64):
        for i in range(3):
            acc = -w * w * th[i] - 2 * z * w * vel[i]
            vel[i] += acc * dt
            th[i] += vel[i] * dt + math.radians(random.uniform(-1.2, 1.2)) * math.exp(-n / 22)
        states.append(tuple(th))
    states[-1] = (0.0, 0.0, 0.0)
    t_ep0, t_ep1 = 0.08, 0.70
    frames = [(0.0, states[0])]
    for n, st in enumerate(states):
        frames.append((t_ep0 + (t_ep1 - t_ep0) * n / (len(states) - 1), st))
    final = states[-1]
    frames += [(0.88, final), (0.92, final), (0.925, states[0]), (1.0, states[0])]

    grids, borders, normals, errs = [], [], [], []
    e0 = math.sqrt(sum(a * a for a in states[0]))
    for t, st in frames:
        g, b, nrm = plane_paths(rot(*st))
        grids.append((t, g))
        borders.append((t, b))
        normals.append((t, nrm))
        errs.append((t, f(160 * math.sqrt(sum(a * a for a in st)) / e0)))

    vis = [(0, 1), (0.88, 1), (0.92, 0), (0.95, 0), (0.99, 1), (1, 1)]
    aligned = fades([(0.70, 0.88)], ramp=0.02)

    # episode return curve (training progress across many episodes)
    rng = random.Random(7)
    ret = []
    for i in range(60):
        v = 1 - math.exp(-i / 16) + rng.uniform(-0.09, 0.09) * math.exp(-i / 40)
        ret.append(f"{'M' if i == 0 else 'L'}{f(420 + i * 2.6)} {f(118 - 50 * max(0, min(1, v)))}")

    body = f"""
<path d="{t_border}" fill="none" stroke="{ACCENT}" stroke-width="1.3" stroke-dasharray="4 4"/>
<path d="{t_normal}" stroke="{ACCENT}" stroke-width="1.3" stroke-dasharray="3 3"/>
<circle cx="{CX}" cy="{f(proj((0, 0, 0))[1])}" r="2.5" fill="{ACCENT}"/>
<clipPath id="stage"><rect x="0" y="46" width="404" height="226"/></clipPath>
<g opacity="1" clip-path="url(#stage)">
  {anim("opacity", vis, T)}
  <path d="{grids[0][1]}" fill="none" stroke="{INK}" stroke-opacity="0.3" stroke-width="1">{anim("d", grids, T)}</path>
  <path d="{borders[0][1]}" fill="{INK}" fill-opacity="0.05" stroke="{INK}" stroke-width="1.8" stroke-linejoin="round">{anim("d", borders, T)}
    {anim("stroke", [(t, ACCENT if v else INK) for t, v in windows([(0.70, 0.88)])], T, discrete=True)}</path>
  <path d="{normals[0][1]}" stroke="{INK}" stroke-width="1.8" stroke-linecap="round">{anim("d", normals, T)}</path>
</g>
<text x="420" y="62" font-size="12" fill="{MUTED}">episode return</text>
<path d="{" ".join(ret)}" fill="none" stroke="{INK}" stroke-width="1.4" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">
  {anim("stroke-dashoffset", [(0, 1), (0.85, 0), (0.92, 0), (0.93, 1), (1, 1)], T)}
</path>
<line x1="420" y1="120" x2="576" y2="120" stroke="{DIM}"/>
<text x="420" y="200" font-size="12" fill="{MUTED}">misalignment</text>
<rect x="420" y="208" width="156" height="6" rx="3" fill="{FAINT}"/>
<rect x="420" y="208" width="156" height="6" rx="3" fill="{ACCENT}">{anim("width", [(t, min(156, float(v))) for t, v in errs], T)}</rect>
<text x="420" y="238" font-size="12" fill="{ACCENT}" opacity="0">aligned{anim("opacity", aligned, T)}</text>
"""
    write("card-alignment.svg", card(
        "05", "ALIGNMENT", "ASELSAN · 2025",
        "Teaching an agent to align a 3D plane",
        "reinforcement learning · Gymnasium · Python",
        body))


# ---------------------------------------------------------------------------
# 03 · TIMBER: three anomaly-detection models scan the same wooden board
# ---------------------------------------------------------------------------
def timber():
    random.seed(3)
    T = 12
    BX0, BY0, BX1, BY1 = 30, 56, 570, 240
    KX, KY = 172, 150  # knot: natural texture, should stay "normal"
    CX, CY = 418, 118  # crack: the real defect

    lines = []
    n = 30
    for i in range(n):
        yb = BY0 + 6 + i * (BY1 - BY0 - 12) / (n - 1)
        pts = []
        for x in range(BX0, BX1 + 1, 6):
            y = yb + 2.0 * math.sin(x / 61 + i * 0.37) + 1.1 * math.sin(x / 23 + i * 1.7)
            dy = yb - KY
            bump = 26 * math.exp(-((x - KX) ** 2) / (2 * 52 ** 2)) * math.exp(-(dy ** 2) / (2 * 34 ** 2))
            y += bump * (1 if dy >= 0 else -1)
            pts.append(f"{'M' if not pts else 'L'}{x} {f(y)}")
        lines.append(f'<path d="{" ".join(pts)}" stroke-opacity="{f(0.16 + 0.22 * random.random(), 2)}"/>')

    knot = "".join(
        f'<ellipse cx="{KX}" cy="{KY}" rx="{f(r * 1.5)}" ry="{f(r)}" stroke-opacity="{f(0.65 - r / 40, 2)}"/>'
        for r in (4, 8, 12, 16)
    )
    crack_pts = [(CX - 46, CY - 6)]
    for k in range(1, 12):
        crack_pts.append((CX - 46 + k * 8.5, CY - 6 + k * 1.6 + random.uniform(-3.5, 3.5)))
    crack = "M" + "L".join(f"{f(x)} {f(y)}" for x, y in crack_pts)

    passes = [(0.02, 0.22, 0.30), (0.35, 0.55, 0.63), (0.68, 0.88, 0.96)]
    models = ["GLASS", "EfficientAD", "INP-Former"]
    false_pos = [0.32, 0.16, 0.07]  # how much each map lights up the knot

    defs = [
        f'<radialGradient id="hot"><stop offset="0" stop-color="{ACCENT}" stop-opacity="0.85"/>'
        f'<stop offset="0.55" stop-color="{ACCENT}" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/></radialGradient>'
    ]
    maps, scanners, labels = [], [], []
    W = BX1 - BX0
    for m, ((s, e, h), name) in enumerate(zip(passes, models)):
        defs.append(
            f'<clipPath id="reveal{m}"><rect x="{BX0}" y="{BY0}" height="{BY1 - BY0}" width="0">'
            f'{anim("width", [(0, 0), (s, 0), (e, W), (1, W)], T, spline=EASE)}</rect></clipPath>'
        )
        rng = random.Random(m + 10)
        speck = "".join(
            f'<ellipse cx="{f(rng.uniform(BX0 + 20, BX1 - 20))}" cy="{f(rng.uniform(BY0 + 15, BY1 - 15))}" '
            f'rx="{f(rng.uniform(10, 22))}" ry="{f(rng.uniform(6, 12))}" opacity="{f(rng.uniform(0.08, 0.2) * (1.4 - m * 0.4), 2)}"/>'
            for _ in range(5)
        )
        maps.append(
            f'<g clip-path="url(#reveal{m})" opacity="0">{anim("opacity", fades([(s, h)], ramp=0.025), T)}'
            f'<g fill="url(#hot)"><ellipse cx="{CX}" cy="{CY + 3}" rx="78" ry="34"/>'
            f'<ellipse cx="{KX}" cy="{KY}" rx="44" ry="30" opacity="{false_pos[m]}"/>{speck}</g></g>'
        )
        scanners.append(
            f'<g opacity="0">{anim("opacity", fades([(s, e)], ramp=0.01), T)}'
            f'{anim_tr("translate", [(0, "0 0"), (s, "0 0"), (e, f"{W} 0"), (1, f"{W} 0")], T, spline=EASE)}'
            f'<rect x="{BX0 - 18}" y="{BY0}" width="18" height="{BY1 - BY0}" fill="url(#scanfade)"/>'
            f'<line x1="{BX0}" y1="{BY0}" x2="{BX0}" y2="{BY1}" stroke="{INK}" stroke-width="1.5"/></g>'
        )
        active = [(0, DIM)] + [(t, INK if v else DIM) for t, v in windows([(s, h + 0.03)])][1:]
        labels.append((name, active))
    defs.append(
        f'<linearGradient id="scanfade"><stop offset="0" stop-color="{INK}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{INK}" stop-opacity="0.16"/></linearGradient>'
    )
    outline_spans = [(e + 0.02, h + 0.03) for s, e, h in passes]
    label_svg, x = [], BX1
    for name, active in reversed(labels):
        label_svg.append(
            f'<text x="{x}" y="262" font-size="12" fill="{DIM}" text-anchor="end">{name}'
            f'{anim("fill", active, T, discrete=True)}</text>'
        )
        x -= len(name) * 7.4 + 20

    body = f"""
<rect x="{BX0}" y="{BY0}" width="{BX1 - BX0}" height="{BY1 - BY0}" rx="4" fill="#17140f"/>
<g fill="none" stroke="{SAND}" stroke-width="1.1">{"".join(lines)}{knot}</g>
<ellipse cx="{KX}" cy="{KY}" rx="3.5" ry="2.4" fill="#0b0907"/>
<path d="{crack}" fill="none" stroke="#070605" stroke-width="2.6" stroke-linejoin="bevel"/>
<path d="{crack}" fill="none" stroke="{SAND}" stroke-opacity="0.45" stroke-width="0.8" transform="translate(0 1.6)"/>
{"".join(maps)}
<rect x="{CX - 60}" y="{CY - 26}" width="120" height="56" rx="4" fill="none" stroke="{ACCENT}" stroke-width="1.4" stroke-dasharray="5 4" opacity="0">
  {anim("opacity", fades(outline_spans, ramp=0.015), T)}
</rect>
{"".join(scanners)}
<text x="{BX0}" y="262" font-size="12" fill="{MUTED}">anomaly map</text>
{"".join(label_svg)}
"""
    write("card-timber.svg", card(
        "06", "TIMBER", "3 models · 1 board",
        "Wood surface anomaly detection",
        "GLASS · EfficientAD · INP-Former · Flask · React",
        body, extra_defs="".join(defs)))


# ---------------------------------------------------------------------------
# 04 · LEAVES: a leaf is drawn, scanned, and its early spots are found
# ---------------------------------------------------------------------------
def leaves():
    random.seed(4)
    T = 10
    LX, LY, ROT, SC = 190, 160, -38, 0.92

    def to_screen(x, y):
        a = math.radians(ROT)
        x, y = x * SC, y * SC
        return LX + x * math.cos(a) - y * math.sin(a), LY + x * math.sin(a) + y * math.cos(a)

    def half_width(y):  # leaf half-width at local y (base at +100, tip at -100)
        t = (100 - y) / 200
        return 70 * math.sin(math.pi * t) ** 0.9 * (1 - 0.25 * t)

    edge_r = [(half_width(y), y) for y in range(100, -101, -5)]
    outline_local = edge_r + [(-x, y) for x, y in reversed(edge_r)]
    outline = "M" + "L".join(f"{f(sx)} {f(sy)}" for sx, sy in (to_screen(x, y) for x, y in outline_local)) + "Z"
    sx0, sy0 = to_screen(0, 100)
    sx1, sy1 = to_screen(0, 128)
    stem = f"M{f(sx0)} {f(sy0)}L{f(sx1)} {f(sy1)}"
    mid = "M" + "L".join(f"{f(a)} {f(b)}" for a, b in (to_screen(0, y) for y in range(100, -96, -15)))
    veins = []
    for y in range(72, -72, -24):
        for side in (1, -1):
            hw = half_width(y - 22) * 0.86
            p0, p1, p2 = to_screen(0, y), to_screen(side * hw * 0.45, y - 8), to_screen(side * hw, y - 24)
            veins.append(f"M{f(p0[0])} {f(p0[1])}Q{f(p1[0])} {f(p1[1])} {f(p2[0])} {f(p2[1])}")

    spots_local = [(26, 34, 9), (-30, 6, 7), (20, -30, 8), (-18, -58, 6), (36, 58, 5), (-36, 46, 6)]
    scan_y0, scan_y1, t_s0, t_s1 = 58, 264, 0.34, 0.66
    spots, marks, det_times = [], [], []
    for x, y, r in spots_local:
        cx, cy = to_screen(x, y)
        td = t_s0 + (t_s1 - t_s0) * (cy - scan_y0) / (scan_y1 - scan_y0)
        det_times.append(td)
        blob = []
        for k in range(10):
            a = 2 * math.pi * k / 10
            rr = r * random.uniform(0.75, 1.2)
            blob.append(f"{f(cx + rr * math.cos(a))} {f(cy + rr * math.sin(a))}")
        spots.append(
            f'<path d="M{"L".join(blob)}Z" fill="{ACCENT}" fill-opacity="0.55" opacity="0">'
            f'{anim("opacity", fades([(0.26 + random.uniform(0, 0.05), 0.9)], ramp=0.06), T)}</path>'
        )
        b = r + 7
        corners = (f"M{f(cx - b)} {f(cy - b + 5)}V{f(cy - b)}H{f(cx - b + 5)}"
                   f"M{f(cx + b - 5)} {f(cy - b)}H{f(cx + b)}V{f(cy - b + 5)}"
                   f"M{f(cx + b)} {f(cy + b - 5)}V{f(cy + b)}H{f(cx + b - 5)}"
                   f"M{f(cx - b + 5)} {f(cy + b)}H{f(cx - b)}V{f(cy + b - 5)}")
        marks.append(
            f'<path d="{corners}" fill="none" stroke="{ACCENT}" stroke-width="1.4" opacity="0">'
            f'{anim("opacity", fades([(td, 0.9)], ramp=0.015), T)}</path>'
        )

    # probability of disease rises with each detection
    det_sorted = sorted(det_times)
    p = 0.07
    pf = [(0, 0.07), (det_sorted[0] - 0.005, 0.07)]
    for td in det_sorted:
        p = p + (0.88 - p) * 0.42
        pf.append((td + 0.02, p))
    pf += [(0.9, p), (0.95, 0.07), (1, 0.07)]
    BAR = 200
    bar = [(t, f(BAR * v)) for t, v in pf]
    alarm = det_sorted[2]
    sens_pts = []
    rng = random.Random(9)
    for i in range(0, 201, 4):
        drift = 0 if i < 110 else (i - 110) * 0.28
        sens_pts.append(f"{'M' if not sens_pts else 'L'}{376 + i} {f(126 - drift - 9 * math.sin(i / 13) - rng.uniform(-4, 4))}")

    body = f"""
<g opacity="1">
  {anim("opacity", [(0, 1), (0.9, 1), (0.96, 0), (1, 0)], T)}
  <path d="{outline}" fill="{INK}" fill-opacity="0" stroke="{INK}" stroke-width="1.6" stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">
    {anim("stroke-dashoffset", [(0, 1), (0.2, 0), (1, 0)], T, spline=EASE)}
    {anim("fill-opacity", [(0, 0), (0.14, 0), (0.24, 0.06), (1, 0.06)], T)}
  </path>
  <path d="{stem}" stroke="{INK}" stroke-width="1.6" stroke-linecap="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">
    {anim("stroke-dashoffset", [(0, 1), (0.02, 1), (0.08, 0), (1, 0)], T)}
  </path>
  <path d="{mid}" fill="none" stroke="{MUTED}" stroke-width="1.2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">
    {anim("stroke-dashoffset", [(0, 1), (0.12, 1), (0.24, 0), (1, 0)], T, spline=EASE)}
  </path>
  <g fill="none" stroke="{MUTED}" stroke-opacity="0.7" stroke-width="0.9">
    {"".join(f'<path d="{v}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">{anim("stroke-dashoffset", [(0, 1), (0.16 + i * 0.006, 1), (0.28 + i * 0.006, 0), (1, 0)], T, spline=EASE)}</path>' for i, v in enumerate(veins))}
  </g>
  {"".join(spots)}
  {"".join(marks)}
  <g opacity="0">
    {anim("opacity", fades([(t_s0, t_s1)], ramp=0.015), T)}
    {anim_tr("translate", [(0, "0 0"), (t_s0, "0 0"), (t_s1, f"0 {scan_y1 - scan_y0}"), (1, f"0 {scan_y1 - scan_y0}")], T)}
    <line x1="70" y1="{scan_y0}" x2="320" y2="{scan_y0}" stroke="{INK}" stroke-width="1.2" stroke-opacity="0.8"/>
    <rect x="70" y="{scan_y0 - 16}" width="250" height="16" fill="url(#scanv)"/>
  </g>
  <text x="376" y="72" font-size="12" fill="{MUTED}">sensor stream</text>
  <path d="{" ".join(sens_pts)}" fill="none" stroke="{INK}" stroke-width="1.3" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">
    {anim("stroke-dashoffset", [(0, 1), (0.05, 1), (0.7, 0), (1, 0)], T)}
  </path>
  <text x="376" y="176" font-size="12" fill="{MUTED}">p(disease)</text>
  <rect x="376" y="186" width="{BAR}" height="6" rx="3" fill="{FAINT}"/>
  <rect x="376" y="186" width="{bar[0][1]}" height="6" rx="3" fill="{INK}">
    {anim("width", bar, T, spline=EASE)}
    {anim("fill", [(t, ACCENT if v else INK) for t, v in windows([(alarm, 0.95)])], T, discrete=True)}
  </rect>
  <text x="376" y="222" font-size="12" fill="{INK}">healthy{anim("opacity", windows([(0, alarm)]), T, discrete=True)}</text>
  <text x="376" y="222" font-size="12" fill="{ACCENT}" opacity="0">early signs{anim("opacity", windows([(alarm, 0.95)]), T, discrete=True)}</text>
</g>
"""
    defs = (f'<linearGradient id="scanv" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{INK}" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="{INK}" stop-opacity="0.12"/></linearGradient>')
    write("card-leaves.svg", card(
        "07", "LEAVES", "TÜBİTAK · research",
        "Early plant disease detection",
        "image + sensor data · deep learning",
        body, extra_defs=defs))


# ---------------------------------------------------------------------------
# 05 · KIDNEYS: NAIM reads incomplete clinical rows without imputing them
# ---------------------------------------------------------------------------
def kidneys():
    random.seed(5)
    T = 10
    feats = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc"]
    miss_rate = [0.02, 0.03, 0.12, 0.11, 0.12, 0.38, 0.16, 0.01, 0.01, 0.11, 0.05, 0.04]
    NRW, CWD, CHT, PX, PY = 8, 26, 15, 30, 20
    X0, Y0 = 34, 72
    starts = [0.05 + k * 0.1 for k in range(NRW)]
    win_len = 0.085

    head = "".join(
        f'<text x="{X0 + c * PX + CWD / 2}" y="{Y0 - 8}" font-size="10.5" fill="{DIM}" text-anchor="middle">{name}</text>'
        for c, name in enumerate(feats)
    )
    cells, masks, outs, links = [], [], [], []
    ox = X0 + len(feats) * PX + 46
    for r in range(NRW):
        s = starts[r]
        y = Y0 + r * PY
        missing_any = False
        for c in range(len(feats)):
            x = X0 + c * PX
            missing = random.random() < miss_rate[c] * 1.4
            if c == 5 and r in (1, 4):
                missing = True
            if missing:
                missing_any = True
                cells.append(f'<rect x="{x + 0.5}" y="{y + 0.5}" width="{CWD - 1}" height="{CHT - 1}" rx="2" fill="none" stroke="{DIM}" stroke-dasharray="2 2"/>')
                masks.append(
                    f'<path d="M{x + 9} {y + 4}l8 7m0 -7l-8 7" stroke="{ACCENT}" stroke-width="1.4" opacity="0">'
                    f'{anim("opacity", [(0, 0), (s, 0), (s + 0.02, 1), (s + win_len, 1), (s + win_len + 0.02, 0.55), (0.9, 0.55), (0.95, 0), (1, 0)], T)}</path>'
                )
            else:
                base = f(0.12 + 0.18 * random.random(), 2)
                cells.append(
                    f'<rect x="{x}" y="{y}" width="{CWD}" height="{CHT}" rx="2" fill="{INK}" opacity="{base}">'
                    f'{anim("opacity", [(0, base), (s, base), (s + 0.02, 0.85), (s + win_len, 0.85), (s + win_len + 0.02, 0.42), (0.9, 0.42), (0.95, base), (1, base)], T)}</rect>'
                )
        ckd = random.random() < 0.62
        cy = y + CHT / 2
        t_out = s + 0.05
        links.append(
            f'<path d="M{X0 + len(feats) * PX + 2} {cy}H{ox - 12}" stroke="{MUTED}" stroke-width="1" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">'
            f'{anim("stroke-dashoffset", [(0, 1), (s + 0.02, 1), (t_out, 0), (0.9, 0), (0.95, 1), (1, 1)], T)}</path>'
        )
        fill = f'fill="{ACCENT}"' if ckd else f'fill="none" stroke="{INK}" stroke-width="1.4"'
        outs.append(
            f'<circle cx="{ox}" cy="{cy}" r="5.5" {fill} opacity="0">'
            f'{anim("opacity", [(0, 0), (t_out, 0), (t_out + 0.015, 1), (0.9, 1), (0.95, 0), (1, 0)], T)}</circle>'
        )
    sel = [(0, f"0 {0}")]
    for r, s in enumerate(starts):
        sel += [(s, f"0 {r * PY}"), (s + win_len, f"0 {r * PY}")]
    sel += [(0.9, f"0 {(NRW - 1) * PY}"), (0.96, "0 0"), (1, "0 0")]

    body = f"""
{head}
<text x="{ox}" y="{Y0 - 8}" font-size="10.5" fill="{DIM}" text-anchor="middle">ckd?</text>
<rect x="{X0 - 6}" y="{Y0 - 3}" width="{len(feats) * PX + 8}" height="{CHT + 6}" rx="4" fill="{FAINT}" stroke="{DIM}" stroke-opacity="0.6">
  {anim_tr("translate", sel, T, spline=EASE)}
</rect>
{"".join(cells)}
{"".join(masks)}
{"".join(links)}
{"".join(outs)}
<g font-size="12" fill="{MUTED}">
  <rect x="{X0}" y="246" width="14" height="10" rx="2" fill="none" stroke="{DIM}" stroke-dasharray="2 2"/>
  <text x="{X0 + 22}" y="255">missing value → masked, not imputed</text>
  <circle cx="{X0 + 318}" cy="251" r="4.5" fill="{ACCENT}"/>
  <text x="{X0 + 328}" y="255">ckd</text>
  <circle cx="{X0 + 368}" cy="251" r="4.5" fill="none" stroke="{INK}" stroke-width="1.3"/>
  <text x="{X0 + 378}" y="255">not ckd</text>
</g>
"""
    write("card-kidneys.svg", card(
        "08", "KIDNEYS", "NAIM · UCI CKD",
        "Learning from incomplete clinical data",
        "NAIM vs. SVM · KNN · Random Forest",
        body))


# ---------------------------------------------------------------------------
# 06 · METAL: a 4x4 keypad, a bouncing pin and a SysTick-timed debounce
# ---------------------------------------------------------------------------
def metal():
    random.seed(6)
    T = 9
    KEYS = ["123A", "456B", "789C", "*0#D"]
    KX, KY, KW, KH, KP, KQ = 34, 66, 34, 27, 40, 33
    presses = [((1, 0), 0.10), ((0, 1), 0.42), ((2, 3), 0.70)]
    HOLD, SETTLE = 0.15, 0.035
    TX0, TX1 = 236, 576
    TW = TX1 - TX0
    X = lambda t: TX0 + t * TW

    keys = []
    for r, row in enumerate(KEYS):
        for c, ch in enumerate(row):
            x, y = KX + c * KP, KY + r * KQ
            spans = [(t, t + HOLD) for (pr, pc), t in presses if (pr, pc) == (r, c)]
            fill_anim = stroke_anim = txt_anim = ""
            if spans:
                fill_anim = anim("fill-opacity", [(v[0], 0.9 if v[1] else 0) for v in windows(spans)], T, discrete=True)
                txt_anim = anim("fill", [(v[0], BG if v[1] else MUTED) for v in windows(spans)], T, discrete=True)
            keys.append(
                f'<rect x="{x}" y="{y}" width="{KW}" height="{KH}" rx="4" fill="{ACCENT}" fill-opacity="0" stroke="{DIM}">{fill_anim}</rect>'
                f'<text x="{x + KW / 2}" y="{y + 18}" font-size="12" fill="{MUTED}" text-anchor="middle">{ch}{txt_anim}</text>'
            )

    # raw (active-low, bouncing) and debounced traces over one loop
    HI1, LO1, HI2, LO2 = 84, 106, 144, 166
    raw = [f"M{TX0} {HI1}"]
    clean = [f"M{TX0} {HI2}"]
    for _, t in presses:
        for edge_t, to_low in ((t, True), (t + HOLD, False)):
            x = X(edge_t)
            seq = [to_low, not to_low, to_low, not to_low, to_low]
            for k, low in enumerate(seq):
                xx = x + k * random.uniform(1.4, 3.2)
                raw.append(f"H{f(xx)}V{LO1 if low else HI1}")
            clean.append(f"H{f(X(edge_t + SETTLE))}V{LO2 if to_low else HI2}")
    raw.append(f"H{TX1}")
    clean.append(f"H{TX1}")
    ticks = "".join(f'<line x1="{f(X(i / 68))}" y1="196" x2="{f(X(i / 68))}" y2="{202 if i % 4 else 206}"/>' for i in range(69))

    # debounce state machine label
    states = []
    for _, t in presses:
        states += [("BOUNCE", t, t + SETTLE), ("PRESSED", t + SETTLE, t + HOLD), ("BOUNCE", t + HOLD, t + HOLD + SETTLE)]
    idle = [(0, presses[0][1])] + [
        (presses[i][1] + HOLD + SETTLE, presses[i + 1][1]) for i in range(len(presses) - 1)
    ] + [(presses[-1][1] + HOLD + SETTLE, 1)]
    state_svg = [f'<text x="{TX0 + 48}" y="234" font-size="12.5" fill="{MUTED}">IDLE{anim("opacity", windows(idle), T, discrete=True)}</text>']
    for name, color in (("BOUNCE", INK), ("PRESSED", ACCENT)):
        spans = [(s, e) for n, s, e in states if n == name]
        state_svg.append(f'<text x="{TX0 + 48}" y="234" font-size="12.5" fill="{color}" opacity="0">{name}{anim("opacity", windows(spans), T, discrete=True)}</text>')

    # row/column lines seen by the MCU (after debounce)
    bits = []
    for i, lab in enumerate(["R0", "R1", "R2", "R3", "C0", "C1", "C2", "C3"]):
        x = KX + i * 19
        spans = [(t + SETTLE, t + HOLD) for (pr, pc), t in presses if (i < 4 and pr == i) or (i >= 4 and pc == i - 4)]
        a = anim("fill-opacity", [(v[0], 0.95 if v[1] else 0) for v in windows(spans)], T, discrete=True) if spans else ""
        bits.append(f'<rect x="{x}" y="214" width="14" height="14" rx="2" fill="{ACCENT}" fill-opacity="0" stroke="{DIM}">{a}</rect>'
                    f'<text x="{x + 7}" y="242" font-size="9.5" fill="{DIM}" text-anchor="middle">{lab}</text>')

    body = f"""
{"".join(keys)}
{"".join(bits)}
<text x="{KX}" y="258" font-size="11.5" fill="{DIM}">row / col lines after debounce</text>
<clipPath id="draw"><rect x="{TX0 - 2}" y="60" height="160" width="0">{anim("width", [(0, 0), (1, TW + 4)], T)}</rect></clipPath>
<text x="{TX0}" y="68" font-size="12" fill="{MUTED}">raw pin (active low)</text>
<text x="{TX0}" y="130" font-size="12" fill="{MUTED}">debounced</text>
<g fill="none" stroke-linejoin="miter">
  <path d="{"".join(raw)}" stroke="{DIM}" stroke-width="1.2"/>
  <path d="{"".join(clean)}" stroke="{FAINT}" stroke-width="1.6"/>
  <g clip-path="url(#draw)">
    <path d="{"".join(raw)}" stroke="{INK}" stroke-width="1.2"/>
    <path d="{"".join(clean)}" stroke="{ACCENT}" stroke-width="1.8"/>
  </g>
</g>
<g stroke="{DIM}" stroke-width="1">{ticks}</g>
<text x="{TX0}" y="222" font-size="11.5" fill="{DIM}">SysTick · 1 kHz</text>
<text x="{TX0}" y="234" font-size="12.5" fill="{DIM}">state:</text>
{"".join(state_svg)}
<line x1="{TX0}" y1="62" x2="{TX0}" y2="208" stroke="{INK}" stroke-opacity="0.5">
  {anim_tr("translate", [(0, "0 0"), (1, f"{TW} 0")], T)}
</line>
"""
    write("card-metal.svg", card(
        "01", "METAL", "C · Cortex-M4 · no HAL",
        "Bare-metal drivers for STM32F411",
        "register-level GPIO · EXTI/NVIC · SysTick debounce",
        body))


# ---------------------------------------------------------------------------
# 02 · RASPBERRY PI: four processes, three ports, real pins
# ---------------------------------------------------------------------------
def raspi():
    T = 13
    PILL_H = 17

    def box(x0, y0, x1, y1, name, port=None):
        s = (f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="6" fill="{FAINT}" stroke="{DIM}"/>'
             f'<text x="{x0 + 10}" y="{y0 + 18}" font-size="12.5" fill="{INK}">{name}</text>')
        if port:
            s += f'<text x="{x1 - 10}" y="{y0 + 18}" font-size="12" fill="{MUTED}" text-anchor="end">{port}</text>'
        return s

    def polyline_frames(pts, s, e):
        segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        total = sum(segs)
        frames = [(0, f"{pts[0][0]} {pts[0][1]}"), (s, f"{pts[0][0]} {pts[0][1]}")]
        acc = 0
        for i, L in enumerate(segs):
            acc += L
            frames.append((s + (e - s) * acc / total, f"{pts[i + 1][0]} {pts[i + 1][1]}"))
        frames.append((1, frames[-1][1]))
        return frames

    def packet(color, pts, s, e):
        return (f'<g opacity="0">{anim("opacity", windows([(s - 0.004, e + 0.004)]), T, discrete=True)}'
                f'{anim_tr("translate", polyline_frames(pts, s, e), T)}'
                f'<circle r="7" fill="{color}" opacity="0.18"/><circle r="3.6" fill="{color}"/></g>')

    C_OUT = [(170, 160), (214, 160)]
    TO_D = [(324, 154), (352, 154), (352, 90), (380, 90)]
    TO_E = [(324, 166), (352, 166), (352, 196), (380, 196)]
    exchanges = [("relay 1:", "relay ok:", "D", "relay"), ("temp:", "temp 25.3:", "E", "uart"),
                 ("key:", "key 7:", "D", "keypad")]
    starts = [0.04 + 0.31 * k for k in range(3)]

    packets, term, node_txt, hw_anims = [], [], [], {"relay": [], "uart": [], "keypad": []}
    server_flash = []
    for k, ((cmd, reply, node, hw), s) in enumerate(zip(exchanges, starts)):
        route = TO_D if node == "D" else TO_E
        packets.append(packet(INK, C_OUT, s, s + 0.05))
        packets.append(packet(INK, route, s + 0.06, s + 0.12))
        packets.append(packet(ACCENT, route[::-1], s + 0.17, s + 0.23))
        packets.append(packet(ACCENT, C_OUT[::-1], s + 0.24, s + 0.29))
        ny = 104 if node == "D" else 210
        node_txt.append(f'<text x="550" y="{ny}" font-size="11" fill="{INK}" text-anchor="end" opacity="0">{cmd}'
                        f'{anim("opacity", windows([(s + 0.12, s + 0.17)]), T, discrete=True)}</text>')
        node_txt.append(f'<text x="550" y="{ny}" font-size="11" fill="{ACCENT}" text-anchor="end" opacity="0">{reply}'
                        f'{anim("opacity", windows([(s + 0.17, s + 0.23)]), T, discrete=True)}</text>')
        server_flash += [(s + 0.05, s + 0.06), (s + 0.23, s + 0.24)]
        hw_anims[hw].append((s + 0.12, s + 0.17))
        nxt = starts[k + 1] if k < 2 else 0.97
        term.append(f'<text x="34" y="160" font-size="11.5" fill="{MUTED}" opacity="0">&gt; {cmd}'
                    f'{anim("opacity", windows([(s, nxt)]), T, discrete=True)}</text>')
        term.append(f'<text x="34" y="180" font-size="11.5" fill="{ACCENT}" opacity="0">{reply}'
                    f'{anim("opacity", windows([(s + 0.29, nxt)]), T, discrete=True)}</text>')

    def pill(x0, y0, x1, label, spans=None, latch=False):
        a = ta = ""
        if spans:
            if latch:  # relay stays on until the loop resets
                spans = [(spans[0][0], 0.96)]
            a = anim("fill-opacity", [(t, 0.9 if v else 0) for t, v in windows(spans)], T, discrete=True)
            ta = anim("fill", [(t, BG if v else MUTED) for t, v in windows(spans)], T, discrete=True)
        return (f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="18" rx="4" fill="{ACCENT}" fill-opacity="0" stroke="{DIM}">{a}</rect>'
                f'<text x="{(x0 + x1) / 2}" y="{y0 + 13}" font-size="11" fill="{MUTED}" text-anchor="middle">{label}{ta}</text>')

    flash = anim("stroke", [(t, INK if v else DIM) for t, v in windows(server_flash)], T, discrete=True)
    uart_note = hw_anims["uart"][0]
    body = f"""
<rect x="196" y="50" width="380" height="216" rx="10" fill="none" stroke="{DIM}" stroke-dasharray="4 4"/>
<text x="208" y="68" font-size="11.5" fill="{MUTED}">Raspberry Pi 3</text>
<g stroke="{DIM}" fill="none">
  <path d="M170 160H214"/>
  <path d="M324 154H352V90H380"/>
  <path d="M324 166H352V196H380"/>
  <path d="M470 110V120M470 216V226"/>
</g>
<rect x="24" y="112" width="146" height="96" rx="6" fill="{FAINT}" stroke="{DIM}"/>
<text x="34" y="130" font-size="12.5" fill="{INK}">ClientNode</text>
<text x="160" y="130" font-size="12" fill="{MUTED}" text-anchor="end">PC</text>
<line x1="24" y1="140" x2="170" y2="140" stroke="{DIM}"/>
{"".join(term)}
<rect x="214" y="138" width="110" height="44" rx="6" fill="{FAINT}" stroke="{DIM}">{flash}</rect>
<text x="224" y="156" font-size="12.5" fill="{INK}">ServerNode</text>
<text x="224" y="173" font-size="12" fill="{MUTED}">:7001 router</text>
{box(380, 70, 560, 110, "DigitalIONode", ":7002")}
<text x="390" y="104" font-size="10.5" fill="{DIM}">libgpiod</text>
{pill(380, 120, 436, "GPIO17")}
{pill(442, 120, 498, "relay", hw_anims["relay"], latch=True)}
{pill(504, 120, 560, "keypad", hw_anims["keypad"])}
{box(380, 176, 560, 216, "EnvSensorNode", ":7003")}
<text x="390" y="210" font-size="10.5" fill="{DIM}">pthread</text>
{pill(380, 226, 496, "/dev/serial0", hw_anims["uart"])}
<text x="504" y="239" font-size="11" fill="{ACCENT}" opacity="0">t → 25.3{anim("opacity", windows([uart_note]), T, discrete=True)}</text>
{"".join(node_txt)}
{"".join(packets)}
<text x="24" y="232" font-size="11.5" fill="{DIM}">TCP/IP sockets</text>
<text x="24" y="248" font-size="11.5" fill="{DIM}">C++17 · CMake</text>
"""
    write("card-raspi.svg", card(
        "02", "RASPBERRY PI", "systems programming",
        "Remote control system on a Raspberry Pi",
        "4 processes · TCP sockets · GPIO · UART · C++17",
        body))


# ---------------------------------------------------------------------------
# 04 · SAND: the falling-sand rules from the C++ code, simulated and replayed
# ---------------------------------------------------------------------------
def sand():
    rng = random.Random(2024)
    T = 14
    COLS, ROWS, C = 68, 25, 8
    GX, GY = 28, 64
    SAND_, WATER = 1, 2
    STONE = "S"
    grid = [[None] * COLS for _ in range(ROWS)]
    for y in range(14, 17):
        for x in range(6, 28):
            grid[y][x] = STONE
    for y in range(18, ROWS):
        for x in range(45, 48):
            grid[y][x] = STONE
    kind, pos, hist, born = {}, {}, {}, {}
    step_no = 0

    def typ(x, y):
        v = grid[y][x]
        return 0 if v is None else ("S" if v == STONE else kind[v])

    def move(p, nx, ny):
        x, y = pos[p]
        other = grid[ny][nx]
        grid[ny][nx], grid[y][x] = p, other
        if other is not None:  # swap: sand sinks, water rises into its place
            pos[other] = (x, y)
            hist[other].append((step_no, x, y))
        pos[p] = (nx, ny)
        hist[p].append((step_no, nx, ny))

    def update(p):
        x, y = pos[p]
        if y + 1 == ROWS:
            return
        r = rng.randint(0, 1)
        if kind[p] == SAND_:  # MoveableSolid::update
            if typ(x, y + 1) in (0, WATER):
                return move(p, x, y + 1)
            if r == 0 and x != 0 and typ(x - 1, y + 1) in (0, WATER):
                return move(p, x - 1, y + 1)
            if r == 1 and x + 1 != COLS and typ(x + 1, y + 1) in (0, WATER):
                return move(p, x + 1, y + 1)
        else:  # Liquid::update
            if typ(x, y + 1) == 0:
                return move(p, x, y + 1)
            if r == 0 and x != 0 and typ(x - 1, y) == 0:
                return move(p, x - 1, y)
            if r == 1 and x + 1 != COLS and typ(x + 1, y) == 0:
                return move(p, x + 1, y)

    def spawn(k, cx, n):
        for _ in range(n):
            x, y = cx + rng.randint(-2, 2), rng.randint(0, 2)
            if 0 <= x < COLS and grid[y][x] is None:
                p = len(kind)
                kind[p], pos[p], hist[p], born[p] = k, (x, y), [(step_no, x, y)], step_no
                grid[y][x] = p

    N = 190
    sand_x = lambda i: 10 + (i / 70) * 24 + 2 * math.sin(i / 6) if i < 70 else 55 + 3 * math.sin((i - 118) / 7)
    water_x = lambda i: 60 - (i - 38) * 0.1
    emitters = {"sand": [(0, 70), (118, 166)], "water": [(38, 104)]}  # never at the same time
    snapshots = []  # position of every particle after each step, straight from the grid
    for step_no in range(1, N + 1):
        i = step_no
        if any(a <= i < b for a, b in emitters["sand"]):
            spawn(SAND_, int(round(sand_x(i))), 2 if i < 70 else 1)  # thin stream into the water
        if any(a <= i < b for a, b in emitters["water"]):
            spawn(WATER, int(round(water_x(i))), 2)
        for y in range(ROWS - 2, -1, -1):  # Grid::update order
            for x in range(COLS):
                v = grid[y][x]
                if v is not None and v != STONE:
                    update(v)
        snap = {}
        for y in range(ROWS):
            for x in range(COLS):
                v = grid[y][x]
                if v is not None and v != STONE:
                    snap[v] = (x, y)
        snapshots.append(snap)

    # One sample per particle per step, taken after the whole grid pass. A water
    # cell can move twice in one pass (left-to-right scan), so per-move logging
    # would keep a stale position; the end-of-step grid is the ground truth.
    hist = {p: [(born[p], *snapshots[born[p] - 1][p])] for p in kind}
    for i, snap in enumerate(snapshots, start=1):
        for p, (x, y) in snap.items():
            if i > born[p] and hist[p][-1][1:] != (x, y):
                hist[p].append((i, x, y))

    SIM_END = 0.9
    tt = lambda i: i / N * SIM_END

    def keyframes(h):
        pts = []
        for i, x, y in h:
            if pts and i - 1 > pts[-1][0]:  # hold still until the step before it moves
                pts.append((i - 1, pts[-1][1], pts[-1][2]))
            pts.append((i, x, y))
        out = [pts[0]]
        for k in range(1, len(pts) - 1):
            a, b, c = out[-1], pts[k], pts[k + 1]
            # drop b only if a -> b -> c is one straight, constant-speed move
            if (b[1] - a[1]) * (c[0] - b[0]) != (c[1] - b[1]) * (b[0] - a[0]) or \
               (b[2] - a[2]) * (c[0] - b[0]) != (c[2] - b[2]) * (b[0] - a[0]):
                out.append(b)
        if len(pts) > 1:
            out.append(pts[-1])
        return out

    def replay(kf, i):  # where the SVG draws a particle at step i
        if i <= kf[0][0]:
            return kf[0][1:]
        for a, b in zip(kf, kf[1:]):
            if a[0] <= i <= b[0]:
                r = (i - a[0]) / (b[0] - a[0])
                return (a[1] + (b[1] - a[1]) * r, a[2] + (b[2] - a[2]) * r)
        return kf[-1][1:]

    kfs = {p: keyframes(hist[p]) for p in kind}
    bad = sum(1 for i, snap in enumerate(snapshots, start=1) for p, xy in snap.items()
              if max(abs(u - v) for u, v in zip(replay(kfs[p], i), xy)) > 1e-9)
    assert bad == 0, f"replay differs from simulation in {bad} particle-steps"
    print(f"  sand: {len(kind)} particles, {N} steps, replay matches simulation")

    sand_shades = ["#d8bf8e", "#c9ad7a", "#e2cfa3", "#bfa274"]
    water_shades = ["#5f8ea8", "#6e9db6", "#56829a"]
    parts = []
    for p in kind:
        kf = kfs[p]
        frames = [(0.0, f"{GX + kf[0][1] * C} {GY + kf[0][2] * C}")]
        for i, x, y in kf:
            t = tt(i)
            if t > frames[-1][0]:
                frames.append((t, f"{GX + x * C} {GY + y * C}"))
        frames.append((1.0, frames[-1][1]))
        shade = rng.choice(sand_shades if kind[p] == SAND_ else water_shades)
        vis = anim("opacity", [(0, 0), (tt(born[p]), 1), (1, 1)], T, discrete=True) if born[p] else ""
        parts.append(f'<rect width="7" height="7" fill="{shade}" opacity="{0 if born[p] else 1}">'
                     f'{anim_tr("translate", frames, T)}{vis}</rect>')

    stones = "".join(
        f'<rect x="{GX + x * C}" y="{GY + y * C}" width="7" height="7" fill="#3b3732"/>'
        for y in range(ROWS) for x in range(COLS) if grid[y][x] == STONE
    )

    def cursor(name, fx, spans):
        frames = [(0.0, f"{GX + fx(spans[0][0]) * C} {GY - 4}")]
        for a, b in spans:
            for i in range(a, b + 1, 5):
                frames.append((tt(i), f"{f(GX + fx(i) * C)} {GY - 4}"))
        frames.append((1.0, frames[-1][1]))
        vis = windows([(tt(a), tt(b)) for a, b in spans])
        return (f'<g opacity="0">{anim("opacity", vis, T, discrete=True)}{anim_tr("translate", frames, T)}'
                f'<path d="M0 0V14L3.6 10.6L6.2 16L8.4 15L5.9 9.8H10.5Z" fill="{INK}" stroke="{BG}" stroke-width="1"/>'
                f'<text x="14" y="12" font-size="11" fill="{INK}">{name}</text></g>')

    body = f"""
<rect x="{GX - 6}" y="44" width="{COLS * C + 11}" height="{ROWS * C + 26}" rx="6" fill="#0c0b0a" stroke="{EDGE}"/>
<text x="{GX + 2}" y="57" font-size="11" fill="{DIM}">Sand Simulator</text>
<g fill="{DIM}"><circle cx="{GX + COLS * C - 22}" cy="53" r="2.5"/><circle cx="{GX + COLS * C - 12}" cy="53" r="2.5"/><circle cx="{GX + COLS * C - 2}" cy="53" r="2.5"/></g>
{stones}
<g>
  {anim("opacity", [(0, 1), (0.93, 1), (0.98, 0), (1, 0)], T)}
  {"".join(parts)}
</g>
{cursor("LMB · sand", sand_x, emitters["sand"])}
{cursor("RMB · water", water_x, emitters["water"])}
"""
    write("card-sand.svg", card(
        "04", "SAND", "C++ · SFML",
        "Falling-sand simulation in C++",
        "cellular automaton · sand · water · stone · one class per element",
        body))


CARDS = {"turbines": turbines, "alignment": alignment, "timber": timber,
         "leaves": leaves, "kidneys": kidneys, "metal": metal,
         "raspi": raspi, "sand": sand}

if __name__ == "__main__":
    for name in sys.argv[1:] or CARDS:
        CARDS[name]()
