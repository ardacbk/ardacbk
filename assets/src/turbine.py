"""Turbine: a vibration signal pipeline, read left to right.

Six of the 28 vibration channels are synthesised as a rotor harmonic series plus
noise (different per channel). Everything lives on one circular buffer of
L columns (4 samples per column), so the scroll is seamless. Once per loop a
bearing-like fault is added to every channel: a new non-harmonic tone plus
broadband noise, under a raised-cosine envelope of about two seconds.

Panes, all sharing one time axis ("now" at each pane's right edge):
  1. the waveforms, slowed down for display (low-passed, one glyph per column);
  2. a magnitude spectrum (real FFT, 128-sample Hann window ending now);
  3. a spectrogram (real STFT, 64-sample window, one hop per column);
  4. a reconstruction-error trace: spectral distance of the STFT frame from the
     healthy baseline, against a dashed threshold (the larger of mean + 4 sigma and twice the mean of the healthy errors).
Whatever is not in the healthy baseline is drawn in the accent colour.

The strip uses the fine grid (about twice the cells), so each waveform gets
three rows, the spectrogram shows more frequency bins as a dense heat map, and
the spectrum has more bars.
"""
import sys

import numpy as np

from engine import FINE, SX, Grid, Movie

COLS, ROWS = 196, 22
FPS = 4
STEP = 2                     # columns scrolled per frame (keeps the file small; same speed as 1 col at 8 fps)
L = 72                       # loop length in columns == frames
SPC = SX                     # samples per column
NS = L * SPC                 # samples in the circular buffer
F0_PERIOD = 32               # rotor period in samples (8 columns) -> 9 cycles per loop
FAULT_AT, FAULT_LEN = 0, 16  # columns
TONE_M = 31                  # fault tone: 31 cycles per loop = 0.108 cycles/sample (3.4 x rotor)
NCH = 6
CH_NAMES = ["ch00", "ch05", "ch11", "ch16", "ch22", "ch27"]
SPEC_CH = 2                  # channel analysed by fft / spectrogram / error

# layout
R0, R1 = 2, 19               # content rows
P1 = (7, 58)                 # waveform columns (labels sit in 2..5)
P2 = (64, 93)
P3 = (99, 140)
P4 = (146, 193)
SEP = (61, 96, 143)
LANE = 3                     # rows per waveform
NB = 18                      # spectrogram bins shown, one per row
NBAR = 15                    # fft bars, two columns each


# ---------------------------------------------------------------- the signal
def fault_envelope():
    """Raised cosine over FAULT_LEN columns, per sample, circular."""
    env = np.zeros(NS)
    n = FAULT_LEN * SPC
    s = FAULT_AT * SPC
    env[s:s + n] = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)
    return env


def bandpass_noise(rng, lo, hi):
    """Circular band-limited noise (so the loop wraps without a seam)."""
    x = rng.standard_normal(NS)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(NS)
    X[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(X, NS)
    return y / y.std()


def synth():
    rng = np.random.default_rng(11)
    t = np.arange(NS)
    env = fault_envelope()
    base = np.array([1.0, 0.62, 0.42, 0.28, 0.2])           # harmonic amplitudes
    healthy, faulty = [], []
    for i in range(NCH):
        amp = base * rng.uniform(0.55, 1.45, base.size)
        ph = rng.uniform(0, 2 * np.pi, base.size)
        x = sum(a * np.sin(2 * np.pi * (k + 1) * (NS // F0_PERIOD) * t / NS + p)
                for k, (a, p) in enumerate(zip(amp, ph)))
        x = x * (1 + 0.08 * np.sin(2 * np.pi * t / NS + rng.uniform(0, 6)))   # slow load wobble
        x = x + rng.uniform(0.1, 0.18) * rng.standard_normal(NS)
        s = x.std()
        healthy.append(x / s)
        fx = env * (rng.uniform(0.9, 1.3) * np.sin(2 * np.pi * TONE_M * t / NS + rng.uniform(0, 6))
                    + 2.2 * bandpass_noise(rng, 0.05, 0.3)) / s
        faulty.append(fx)
    return np.array(healthy), np.array(faulty), env


def display_series(h, f):
    """Slowed-down view for the waveform pane: one value per column. The healthy part is
    low-passed (8-sample mean) so a rotor cycle reads as a calm wave; the fault is only
    averaged over its own column, so its tone and noise survive as a rough ripple."""
    ma8 = sum(np.roll(h, -j) for j in range(-2, 6)) / 8
    return ma8[::SPC] / ma8.std() + 2.0 * f.reshape(L, SPC).mean(1) / h.std()


def window_at(sig, end, n):
    """n samples ending at sample index `end`, circular."""
    return sig[(end - n + 1 + np.arange(n)) % NS]


def col_end(c):
    return c * SPC + SPC - 1


# ---------------------------------------------------------------- analysis
def stft(sig, win=64):
    w = np.ones(win)                  # rectangular: rotor harmonics sit exactly on bins
    out = np.empty((L, win // 2 + 1))
    for c in range(L):
        out[c] = np.abs(np.fft.rfft(window_at(sig, col_end(c), win) * w))
    return out


def spectrum(sig, c, win=128):
    return np.abs(np.fft.rfft(window_at(sig, col_end(c), win) * np.hanning(win)))


def db(x):
    return 20 * np.log10(x + 1e-3)


# ---------------------------------------------------------------- drawing helpers
def trace_glyphs(y, nrows, min_slope):
    """One glyph per column for a curve given in subrows (8 per row, 0 = top).

    Flat rule from hero.crest_glyphs: lower third of a cell -> `_`, middle third -> `-`,
    upper third -> `_` in the cell above. `/` `\\` only where the curve moves at least
    min_slope subrows per column; a steeper move fills the rows it crosses.
    """
    out, n = [], len(y)
    for c in range(n):
        d = (y[min(c + 1, n - 1)] - y[max(c - 1, 0)]) / (2 if 0 < c < n - 1 else 1)
        yc = y[c]
        if abs(d) >= min_slope:
            ch = "\\" if d > 0 else "/"
            r0 = int(np.clip((yc - abs(d) / 2) // 8, 0, nrows - 1))
            r1 = int(np.clip((yc + abs(d) / 2) // 8, 0, nrows - 1))
            out += [(c, r, ch) for r in range(r0, r1 + 1)]
            continue
        r = int(np.clip(yc // 8, 0, nrows - 1))
        f = yc - 8 * r
        if f < 8 / 3:
            r, ch = (r - 1, "_") if r > 0 else (r, "-")
        elif f < 16 / 3:
            ch = "-"
        else:
            ch = "_"
        out.append((c, r, ch))
    return out


def build():
    healthy, faulty, env = synth()
    sig = healthy + faulty
    env_col = env.reshape(L, SPC).mean(1)
    near = lambda back: np.array([env_col[(c - np.arange(back)) % L].max() < 1e-6 for c in range(L)])
    clean_cols, clean32 = near(28), near(33)
    D = np.array([display_series(healthy[i], faulty[i]) for i in range(NCH)])

    # ---- spectra and baselines for the analysed channel
    S = stft(sig[SPEC_CH])                                         # (L, 33) rectangular 64
    Pb = np.log10(S[:, 1:33].reshape(L, 8, 4).mean(2) ** 2 + 1e-6)  # 8 band powers
    Pbase = Pb[clean_cols].mean(0)
    err = np.sqrt(((Pb - Pbase) ** 2).mean(1))                     # spectral distance from healthy
    err = np.convolve(np.r_[err[-8:], err], np.ones(9) / 9, "valid")   # trailing mean, circular
    thr = max(err[clean_cols].mean() + 4 * err[clean_cols].std(), 2 * err[clean_cols].mean())
    print(f"error: healthy {err[clean_cols].mean():.3f}+-{err[clean_cols].std():.3f}, thr {thr:.3f}, peak {err.max():.3f}")

    # spectrogram: bins 0..NB-1, one bin per row
    sdb = db(S[:, :NB])
    base_row = sdb[clean_cols].mean(0)
    lo, hi = np.percentile(sdb, 45), np.percentile(sdb, 99.8)
    # fft: bins 0..21, two bins per bar
    F = np.array([spectrum(sig[SPEC_CH], c)[:2 * NBAR].reshape(NBAR, 2).max(1) for c in range(L)])
    Fh = F[clean32].mean(0)
    flo, fhi = np.percentile(db(F[clean32]), 35), np.percentile(db(F), 99.8)
    fv = lambda x: np.clip((db(x) - flo) / (fhi - flo), 0, 1)
    print(f"fft range {flo:.1f}..{fhi:.1f} dB, spec range {lo:.1f}..{hi:.1f} dB")

    w1 = P1[1] - P1[0] + 1
    w3 = P3[1] - P3[0] + 1
    w4 = P4[1] - P4[0] + 1
    nrows = R1 - R0 + 1
    ramp = " .,:;-~=+*o%#@"
    ramp_cl = ["", "t1", "t1", "t1", "t2", "t2", "t2", "t2", "t3", "t3", "t3", "t4", "t4", "t5"]
    thr_row = R0 + 8
    harmonic_bars = {2, 4, 6, 8, 10, 12, 14}

    frames = []
    for cur in range(0, L, STEP):
        g = Grid(COLS, ROWS)

        # labels
        g.text(P1[0] - 5, 0, "28 channels", "t3")
        g.text(P2[0], 0, "fft", "t3")
        g.text(P2[0] + 4, 0, "426 features", "t2")
        g.text(P3[0], 0, "spectrogram", "t3")
        g.text(P3[0] + 12, 0, CH_NAMES[SPEC_CH], "t2")
        g.text(P4[0], 0, "lstm-ae reconstruction error", "t3")
        for s_ in SEP:
            g.put(s_, (R0 + R1) // 2, ">", "t2")

        # ---- pane 1: slowed-down waveforms, one glyph per column
        cols1 = (cur - w1 + 1 + np.arange(w1)) % L
        e1 = env_col[cols1]
        for i in range(NCH):
            y = LANE * 4 - np.clip(D[i][cols1] * 3.4, -LANE * 4 + 1.5, LANE * 4 - 1.5)
            for c, r, ch in trace_glyphs(y, LANE, 6):
                cls = "a" if e1[c] > 0.3 else "b" if e1[c] > 0.04 else ("t3" if i == SPEC_CH else "t2")
                g.put(P1[0] + c, R0 + LANE * i + r, ch, cls)
            g.text(P1[0] - 5, R0 + LANE * i + 1, CH_NAMES[i], "t3" if i == SPEC_CH else "t2")

        # ---- pane 2: fft of the current window, one bar per two columns
        v, vb = fv(F[cur]), fv(Fh)
        for i in range(NBAR):
            h = int(round(v[i] * nrows * 8))
            hb = int(round(vb[i] * nrows * 8))
            hot = h > hb + 10
            for k in range(nrows):
                full = h - 8 * k                                   # subrows of this cell that are filled
                if full <= 0:
                    break
                ch = "|" if full >= 8 else ("." if full < 3 else ":" if full < 6 else "|")
                if hot and 8 * k + 8 > hb + 6:
                    cls = "a"
                else:
                    cls = "t3" if i in harmonic_bars else "t1"
                g.put(P2[0] + 2 * i, R1 - k, ch, cls)

        # ---- pane 3: spectrogram
        cols3 = (cur - w3 + 1 + np.arange(w3)) % L
        for i, c in enumerate(cols3):
            val = np.clip((sdb[c] - lo) / (hi - lo), 0, 0.999) ** 1.15
            ex = sdb[c] - base_row
            for k in range(NB):
                idx = int(val[k] * len(ramp))
                if idx == 0:
                    continue
                cls = ramp_cl[idx]
                if ex[k] > 15:
                    cls = "a"
                elif ex[k] > 10:
                    cls = "b"
                g.put(P3[0] + i, R1 - k, ramp[idx], cls)

        # ---- pane 4: error trace and threshold
        for c in range(0, w4, 2):
            g.put(P4[0] + c, thr_row, "-", "t2")
        cols4 = (cur - w4 + 1 + np.arange(w4)) % L
        e = err[cols4]
        h_ = np.clip(e / thr, 1e-3, None) ** 1.6
        hh = np.where(h_ <= 1, h_, 1 + 0.6 * np.tanh(np.log(np.maximum(h_, 1)) / 1.5))   # soft ceiling
        y = nrows * 8 - 0.5417 * nrows * 8 * hh
        for c, r, ch in trace_glyphs(y, nrows, 3.5):
            g.put(P4[0] + c, R0 + r, ch, "a" if e[c] > thr else "t3")
        if err[cur] > thr:
            g.text(P4[0], R1 + 1, f"anomaly  err {err[cur] / thr * 0.5:.2f}", "a")
            hk = int(np.clip(y[-1] // 8, 0, nrows - 1))
            g.rtext(P4[1] - 3, max(R0, R0 + hk - 1), "anomaly", "a")
        else:
            g.text(P4[0], R1 + 1, f"nominal  err {err[cur] / thr * 0.5:.2f}", "k")
        for p in (P1, P3, P4):
            g.rtext(p[1], R1 + 1, "now", "t1")
        g.text(P2[0], R1 + 1, "freq ->", "t1")
        frames.append(g)

    m = Movie(COLS, ROWS, pad=(12, 14), cell=FINE, title="Turbine condition monitoring pipeline",
              desc="A signal pipeline read left to right: six of 28 vibration channels as waveforms, a "
                   "magnitude spectrum, a scrolling spectrogram and an LSTM autoencoder reconstruction-error "
                   "trace with a threshold. Once per loop a bearing-like fault appears in every pane at the "
                   "same moment, drawn in red, and the readout flips from nominal to anomaly.")
    m.add(frames, FPS)
    return m


if __name__ == "__main__":
    mv = build()
    if "--preview" in sys.argv:
        mv.preview("turbine.png", times=(0, 1.0, 1.8, 2.8, 4.0, 6.0, 8.9))
        mv.preview("turbine-light.png", times=(1.8, 5.0), theme="light")
    mv.write("turbine.svg")
