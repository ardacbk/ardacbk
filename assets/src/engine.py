"""Text-mode engine behind every animation in this profile.

The pipeline is always the same:

1. a script simulates something (waves, sand, a debouncer, a path tracer...)
   and paints it into a float image, a few sub-samples per character cell;
2. `Matcher` turns that image into characters by picking, for every cell, the
   glyph whose shape is closest to the pixels under it;
3. `Movie` packs the frames into one SVG: cells that never change go into a
   static layer, the rest is drawn per frame and switched with CSS.

The SVGs carry no scripts, fonts or external assets, so GitHub can show them
through a plain <img>. Colours are CSS variables with a light-mode override.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent
PREVIEW = Path(__file__).resolve().parent / "_preview"

FONT_PX = 14          # font-size inside the SVG
CW, RH = 8.4, 17.0    # one character cell, in SVG units
SX, SY = 4, 8         # sub-samples per cell used by the matcher
BASELINE = 0.74       # baseline position inside a cell (fraction of RH)
STRIP = (140, 16)     # full-width strip: 1200 x 300
CARD = (68, 18)       # half-width card: 595 x 334
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

# tone ramp t1 (faintest) .. t5 (ink), a = accent, b = dim accent, k = second hue
THEMES = {
    "dark": {"bg": "#0b0e14", "edge": "#212936", "t1": "#2b3544", "t2": "#465770", "t3": "#6d83a0",
             "t4": "#a7b9cf", "t5": "#f1f5fa", "a": "#ff7b42", "b": "#a8532f", "k": "#56cfc3"},
    "light": {"bg": "#fbfbf9", "edge": "#d9dee5", "t1": "#d5dbe3", "t2": "#a9b5c4", "t3": "#74859a",
              "t4": "#47566a", "t5": "#141c27", "a": "#d9480f", "b": "#eda27f", "k": "#0c8479"},
}
CLASSES = ["t1", "t2", "t3", "t4", "t5", "b", "a", "k"]

_MONO_FILES = ["consola.ttf", "CascadiaMono.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc"]
_BOLD_FILES = ["ariblk.ttf", "seguibl.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]


def font(files, size):
    for name in files:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise SystemExit(f"none of these fonts found: {files}")


def mono(size):
    return font(_MONO_FILES, size)


def bold(size, files=None):
    return font(files or _BOLD_FILES, size)


# ------------------------------------------------------------------ grid
class Grid:
    """A rows x cols sheet of (character, colour class) cells."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.ch = np.full((rows, cols), " ", dtype="<U1")
        self.cl = np.full((rows, cols), "", dtype="<U2")

    def copy(self):
        g = Grid(self.cols, self.rows)
        g.ch[:], g.cl[:] = self.ch, self.cl
        return g

    def put(self, c, r, ch, cls):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.ch[r, c], self.cl[r, c] = ch, (cls if ch != " " else "")

    def text(self, c, r, s, cls, clear=True):
        """Write a string; with clear=True spaces erase what is underneath."""
        for i, ch in enumerate(s):
            if ch != " " or clear:
                self.put(c + i, r, ch, cls)

    def rtext(self, c, r, s, cls, clear=True):
        """Right-aligned: the last character lands on column c."""
        self.text(c - len(s) + 1, r, s, cls, clear)

    def ctext(self, c, r, s, cls, clear=True):
        """Centred on column c."""
        self.text(c - len(s) // 2, r, s, cls, clear)

    def clear(self, c0, r0, c1, r1):
        self.ch[r0:r1 + 1, c0:c1 + 1] = " "
        self.cl[r0:r1 + 1, c0:c1 + 1] = ""

    def box(self, c0, r0, c1, r1, cls, corners="..''"):
        """A frame of - and | with corner characters (top-left, top-right, bottom-left, bottom-right)."""
        for c in range(c0 + 1, c1):
            self.put(c, r0, "-", cls)
            self.put(c, r1, "-", cls)
        for r in range(r0 + 1, r1):
            self.put(c0, r, "|", cls)
            self.put(c1, r, "|", cls)
        for (c, r), ch in zip(((c0, r0), (c1, r0), (c0, r1), (c1, r1)), corners):
            self.put(c, r, ch, cls)

    def cover(self, c0, r0, c1, r1):
        """Turn the still-empty cells of a box into covers: in an opaque sprite
        they hide whatever the sprites below have drawn there."""
        c0, r0 = max(c0, 0), max(r0, 0)
        box = self.ch[r0:r1 + 1, c0:c1 + 1]
        self.cl[r0:r1 + 1, c0:c1 + 1][box == " "] = "t1"
        box[box == " "] = NBSP

    def paste(self, chars, classes, c0=0, r0=0, keep_blank=False):
        """Blit arrays of chars/classes; blanks are transparent unless keep_blank."""
        h, w = chars.shape
        dst_ch = self.ch[r0:r0 + h, c0:c0 + w]
        dst_cl = self.cl[r0:r0 + h, c0:c0 + w]
        m = np.ones_like(chars, bool) if keep_blank else chars != " "
        dst_ch[m], dst_cl[m] = chars[m], classes[m]

    def cells(self):
        rr, cc = np.nonzero(self.ch != " ")
        return {(int(r), int(c)): (str(self.ch[r, c]), str(self.cl[r, c])) for r, c in zip(rr, cc)}


# ------------------------------------------------------------------ glyph matcher
NBSP = " "       # a cell that only hides what is underneath (see Grid.cover)
RAMP = " .,:;'`\"^-_~=+*/\\|()<>[]{}!ox#%@"


class Matcher:
    """Image -> characters by nearest glyph shape (SX x SY samples per cell)."""

    def __init__(self, charset=RAMP, sx=SX, sy=SY, gain=1.0):
        self.chars = np.array(list(charset))
        self.sx, self.sy = sx, sy
        k = 6
        w, h = sx * k, sy * k
        f = mono(round(w / 0.55))
        masks = []
        for ch in charset:
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((w / 2, h * BASELINE), ch, font=f, fill=255, anchor="ms")
            a = np.asarray(im, np.float32) / 255
            masks.append(a.reshape(sy, k, sx, k).mean((1, 3)).ravel())
        self.masks = np.array(masks) * gain
        self.norm = (self.masks ** 2).sum(1)

    def match(self, img):
        """img: float array (rows*sy, cols*sx) in 0..1 -> array of chars (rows, cols)."""
        rows, cols = img.shape[0] // self.sy, img.shape[1] // self.sx
        b = img[:rows * self.sy, :cols * self.sx].reshape(rows, self.sy, cols, self.sx)
        b = b.transpose(0, 2, 1, 3).reshape(rows * cols, -1).astype(np.float32)
        d = self.norm[None, :] - 2 * b @ self.masks.T
        return self.chars[d.argmin(1)].reshape(rows, cols)


def cell_mean(img, sx=SX, sy=SY):
    rows, cols = img.shape[0] // sy, img.shape[1] // sx
    return img[:rows * sy, :cols * sx].reshape(rows, sy, cols, sx).mean((1, 3))


def cell_max(img, sx=SX, sy=SY):
    rows, cols = img.shape[0] // sy, img.shape[1] // sx
    return img[:rows * sy, :cols * sx].reshape(rows, sy, cols, sx).max((1, 3))


def tiers(values, edges, names):
    """Map a float array to class names: values < edges[0] -> '', then names[i]."""
    out = np.full(values.shape, "", dtype="<U2")
    for e, n in zip(edges, names):
        out[values >= e] = n
    return out


# ------------------------------------------------------------------ movie
class Sprite:
    """A stack of frames that loops on its own clock."""

    def __init__(self, frames, fps, opaque=False, hold=None):
        self.frames, self.fps, self.opaque = frames, fps, opaque
        self.hold = list(hold) if hold else [1] * len(frames)     # ticks each frame stays up
        self.ticks = sum(self.hold)

    @property
    def dur(self):
        return self.ticks / self.fps

    def at(self, t):
        """The frame showing at time t (seconds)."""
        k = int(t * self.fps + 1e-6) % self.ticks
        for f, h in zip(self.frames, self.hold):
            if k < h:
                return f
            k -= h


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num(x):
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


class Movie:
    def __init__(self, cols, rows, pad=(14, 12), title="", desc=""):
        self.cols, self.rows, self.pad = cols, rows, pad
        self.w = round(cols * CW + 2 * pad[0])
        self.h = round(rows * RH + 2 * pad[1])
        self.title, self.desc = title, desc
        self.sprites, self.under, self.over = [], [], []

    def add(self, frames, fps=12, opaque=False, hold=None):
        """frames: one Grid (static) or a list of Grids.

        hold: optional list, how many ticks (1/fps) each frame stays on screen,
        so a held picture is stored once instead of once per tick.
        """
        if isinstance(frames, Grid):
            frames = [frames]
        self.sprites.append(Sprite(frames, fps, opaque, hold))
        return self

    def x(self, c):
        return self.pad[0] + c * CW

    def y(self, r):
        return self.pad[1] + r * RH

    # -- svg ------------------------------------------------------------
    def _runs(self, cells, opaque):
        """cells {(r, c): (ch, cls)} -> svg text runs (and cover rects if opaque)."""
        rows = {}
        for (r, c), (ch, cls) in cells.items():
            rows.setdefault(r, {}).setdefault(cls, []).append((c, ch))
        out, covers = [], []
        for r in sorted(rows):
            y = _num(self.y(r) + RH * BASELINE)
            if opaque:
                cs = sorted(c for items in rows[r].values() for c, _ in items)
                start = prev = cs[0]
                for c in cs[1:] + [None]:
                    if c is not None and c == prev + 1:
                        prev = c
                        continue
                    w = _num((prev - start + 1) * CW)
                    covers.append(f"M{_num(self.x(start))} {_num(self.y(r))}h{w}v{_num(RH)}h-{w}z")
                    start = prev = c
            for cls in CLASSES:
                items = sorted(it for it in rows[r].get(cls, []) if it[1] != NBSP)
                if not items:
                    continue
                group = [items[0]]
                for it in items[1:] + [None]:
                    if it is not None and it[0] - group[-1][0] <= 40:   # a space is cheaper than a new <text>
                        group.append(it)
                        continue
                    c0, c1 = group[0][0], group[-1][0]
                    chars = [" "] * (c1 - c0 + 1)
                    for c, ch in group:
                        chars[c - c0] = ch
                    n = len(chars)
                    tl = f' textLength="{_num(n * CW)}"' if n > 1 else ""
                    out.append(f'<text class="{cls}" x="{_num(self.x(c0))}" y="{y}"{tl}>{_esc("".join(chars))}</text>')
                    group = [it] if it else []
        cover = f'<path class="cv" d="{"".join(covers)}"/>' if covers else ""
        return cover + "".join(out)

    def svg(self):
        css = [f"text{{font:{FONT_PX}px {MONO};white-space:pre}}",
               "svg{" + ";".join(f"--{k}:{v}" for k, v in THEMES["dark"].items()) + "}",
               "@media (prefers-color-scheme:light){svg{"
               + ";".join(f"--{k}:{v}" for k, v in THEMES["light"].items()) + "}}",
               ".cv{fill:var(--bg)}"]
        css += [f".{c}{{fill:var(--{c})}}" for c in CLASSES]
        body = []
        for si, sp in enumerate(self.sprites):
            cells = [g.cells() for g in sp.frames]
            n = len(cells)
            if n == 1:
                body.append(f"<g>{self._runs(cells[0], sp.opaque)}</g>")
                continue
            keys = set().union(*cells)
            static = {k: cells[0][k] for k in keys if all(k in f and f[k] == cells[0][k] for f in cells)}
            if static:
                body.append(f"<g>{self._runs(static, sp.opaque)}</g>")
            css.append(f".s{si}>g{{visibility:hidden;animation:s{si}h1 {sp.dur:.4g}s linear infinite}}")
            for h in sorted(set(sp.hold)):      # one keyframe rule per distinct hold length
                css.append(f"@keyframes s{si}h{h}{{0%{{visibility:visible}}"
                           f"{100 * h / sp.ticks:.4f}%,100%{{visibility:hidden}}}}")
            layers, tick = [], 0
            for f, h in zip(cells, sp.hold):
                dyn = {k: v for k, v in f.items() if k not in static}
                style = (f"animation-delay:{tick / sp.fps:.4f}s;" if tick else "") +                         (f"animation-name:s{si}h{h}" if h != 1 else "")
                attr = f' style="{style.rstrip(";")}"' if style else ""
                layers.append(f"<g{attr}>{self._runs(dyn, sp.opaque)}</g>" if dyn else f"<g{attr}/>")
                tick += h
            body.append(f'<g class="s{si}">{"".join(layers)}</g>')
        css.append("@media (prefers-reduced-motion:reduce){[class^=s]>g{animation:none}"
                   "[class^=s]>g:first-child{visibility:visible}}")
        w, h = self.w, self.h
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                f'role="img" aria-labelledby="t d" xml:space="preserve">'
                f'<title id="t">{_esc(self.title)}</title><desc id="d">{_esc(self.desc)}</desc>'
                f'<style>{"".join(css)}</style>'
                f'<rect width="{w}" height="{h}" rx="14" fill="var(--bg)"/>'
                f'{"".join(self.under)}{"".join(body)}{"".join(self.over)}'
                f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="14" fill="none" stroke="var(--edge)"/>'
                f'</svg>\n')

    def write(self, name):
        data = self.svg()
        (ASSETS / name).write_text(data, encoding="utf-8")
        print(f"{name}: {len(data.encode('utf-8')) / 1024:.0f} KB, {self.w}x{self.h}, "
              + ", ".join(f"{len(s.frames)}f/{s.dur:g}s" for s in self.sprites))

    # -- png preview (for eyeballing while developing) --------------------
    def frame(self, t):
        """Composite of all sprites at time t (seconds) as one Grid."""
        g = Grid(self.cols, self.rows)
        for sp in self.sprites:
            f = sp.at(t)
            g.paste(f.ch, f.cl)
        g.ch[g.ch == NBSP] = " "
        return g

    def preview(self, name, times=(0,), theme="dark", scale=1):
        PREVIEW.mkdir(exist_ok=True)
        pal = THEMES[theme]
        f = mono(FONT_PX * scale)
        w, h = self.w * scale, self.h * scale
        sheet = Image.new("RGB", (w, h * len(times)), pal["bg"])
        d = ImageDraw.Draw(sheet)
        for i, t in enumerate(times):
            g = self.frame(t)
            for (r, c), (ch, cls) in g.cells().items():
                d.text(((self.x(c) + CW / 2) * scale, i * h + (self.y(r) + RH * BASELINE) * scale), ch,
                       font=f, fill=pal[cls], anchor="ms")
            d.rectangle([0, i * h, w - 1, (i + 1) * h - 1], outline=pal["edge"])
        sheet.save(PREVIEW / name)
        print(f"preview: {PREVIEW / name}")
