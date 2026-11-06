"""Shared palette, type and frame for every animated SVG in this profile.

Each card is a standalone SVG (CSS + SMIL animation, no scripts, no external
assets) so GitHub can render it through a plain <img> tag.
"""
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent

BG = "#121110"
EDGE = "#2a2723"
FAINT = "#211f1c"
DIM = "#4b4740"
MUTED = "#8c867b"
INK = "#ece6da"
SAND = "#c9b48a"
ACCENT = "#ff5a1f"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

CARD_W, CARD_H = 600, 340


def f(x, nd=1):
    """Compact number formatting for path data."""
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


EASE = "0.4 0 0.2 1"


def _kt(frames):
    return ";".join(f(t, 4) for t, _ in frames), ";".join(str(v) for _, v in frames)


def anim(attr, frames, dur, spline=None, discrete=False):
    """<animate> over (t, value) frames, t in 0..1 (first 0, last 1)."""
    kt, vals = _kt(frames)
    mode = 'calcMode="discrete"' if discrete else (
        f'calcMode="spline" keySplines="{";".join([spline] * (len(frames) - 1))}"' if spline else ""
    )
    return (f'<animate attributeName="{attr}" dur="{dur}s" repeatCount="indefinite" '
            f'keyTimes="{kt}" values="{vals}" {mode}/>')


def anim_tr(kind, frames, dur, spline=None):
    kt, vals = _kt(frames)
    mode = f'calcMode="spline" keySplines="{";".join([spline] * (len(frames) - 1))}"' if spline else ""
    return (f'<animateTransform attributeName="transform" type="{kind}" dur="{dur}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{vals}" {mode}/>')


def windows(spans, on=1, off=0):
    """Discrete frames that are `on` inside the given (start, end) spans."""
    frames = [(0.0, off)]
    for s, e in spans:
        if s <= 0:
            frames[0] = (0.0, on)
        else:
            frames.append((s, on))
        if e < 1:
            frames.append((e, off))
    frames.append((1.0, frames[-1][1]))
    return frames


def fades(spans, on=1, off=0, ramp=0.02):
    """Smooth frames that fade to `on` for each (start, end) span."""
    frames = [(0.0, off)]
    for s, e in spans:
        if s <= 0:
            frames[0] = (0.0, on)
        else:
            frames += [(s, off), (min(s + ramp, e), on)]
        if e < 1:
            frames += [(e, on), (min(e + ramp, 1), off)]
    if frames[-1][0] < 1:
        frames.append((1.0, frames[-1][1]))
    return frames


def card(index, tag, right, title, subtitle, body, extra_defs="", extra_css=""):
    """Wrap card artwork in the shared frame (header, footer, border)."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CARD_W} {CARD_H}" width="{CARD_W}" height="{CARD_H}" role="img" aria-label="{esc(title)}">
<defs>
<clipPath id="frame"><rect width="{CARD_W}" height="{CARD_H}" rx="14"/></clipPath>
{extra_defs}
<style>
text {{ font-family: {MONO}; }}
.t-title {{ font-family: {SANS}; font-weight: 600; }}
{extra_css}
</style>
</defs>
<g clip-path="url(#frame)">
<rect width="{CARD_W}" height="{CARD_H}" fill="{BG}"/>
{body}
</g>
<text x="24" y="34" font-size="12.5" fill="{MUTED}" letter-spacing="2">{index}  {esc(tag)}</text>
<text x="{CARD_W - 24}" y="34" font-size="12.5" fill="{DIM}" text-anchor="end">{esc(right)}</text>
<text class="t-title" x="24" y="300" font-size="21" fill="{INK}">{esc(title)}</text>
<text x="24" y="323" font-size="13" fill="{MUTED}">{esc(subtitle)}</text>
<rect x="0.5" y="0.5" width="{CARD_W - 1}" height="{CARD_H - 1}" rx="14" fill="none" stroke="{EDGE}"/>
</svg>
"""


def write(name, svg):
    path = ASSETS / name
    path.write_text(svg, encoding="utf-8")
    print(f"{name}: {len(svg.encode('utf-8')) / 1024:.1f} KB")
