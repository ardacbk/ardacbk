"""STM32F411 keypad: a logic analyser watching one key press.

The real `Keypad_Scan` state machine (IDLE -> DEBOUNCE -> PRESSED, candidate
key, state_enter_tick, debounce_ms = 20) is ported to Python and fed a
simulated bouncing contact. One simulated millisecond is one frame and one
timeline column; the whole 100 ms episode is a ring buffer, so scrolling it
one column per frame loops seamlessly.

Left: the 4x4 keypad (KEY_MAP from keypad.h) with the column that is driven
HIGH highlighted. Middle: contact, the sampled row pin, the FSM state and the
returned key, scrolling right to left ("now" at the right edge). Right: the
registers/variables the firmware would see.

The timeline is drawn like a phosphor screen: the trace is brightest where it
is being written ("now") and fades as it scrolls away, and every edge leaves a
short bright flash. The pressed key sinks into the board by a row.
"""
import sys

from engine import STRIP, Grid, Movie

COLS, ROWS = STRIP
FPS = 10
PERIOD = 100                    # ms of simulated time == frames == ring length
POLL = 2                        # Keypad_Scan is called every 2 ms
DEBOUNCE_MS = 20

KEY_MAP = ["123A", "456B", "789C", "*0#D"]
KEY_R, KEY_C = 1, 1             # the key that gets pressed: '5'
KEY = KEY_MAP[KEY_R][KEY_C]

# ---------------------------------------------------------------- simulation
PRESS_AT, RELEASE_AT = 12, 70
PRESS_BOUNCE = [1, 1, 0, 1, 0, 0, 1, 0, 1]          # chatter after the contact first closes
RELEASE_BOUNCE = [0, 1, 0, 0, 1, 0, 1]              # and after it first opens


def contact_wave():
    c = [0] * PERIOD
    t = PRESS_AT
    for v in PRESS_BOUNCE:
        c[t] = v
        t += 1
    while t < RELEASE_AT:
        c[t] = 1
        t += 1
    for v in RELEASE_BOUNCE:
        c[t] = v
        t += 1
    return c


IDLE, DEBOUNCE, PRESSED = 0, 1, 2
NAMES = ["IDLE", "DEBOUNCE", "PRESSED"]


class Keypad:
    """Faithful port of Keypad_Scan's state machine; raw_scan() is the contact."""

    def __init__(self):
        self.state, self.state_enter_tick, self.candidate_key = IDLE, 0, "\0"
        self.debounce_ms = DEBOUNCE_MS

    def scan(self, raw, systick_ms):
        if self.state == IDLE:
            if raw != "\0":
                self.candidate_key = raw
                self.state_enter_tick = systick_ms
                self.state = DEBOUNCE
        elif self.state == DEBOUNCE:
            if raw != self.candidate_key:
                self.state = IDLE
            elif systick_ms - self.state_enter_tick >= self.debounce_ms:
                self.state = PRESSED
                return self.candidate_key
        elif self.state == PRESSED:
            if raw == "\0":
                self.state = IDLE
        return "\0"


def simulate():
    contact = contact_wave()
    kp = Keypad()
    rec = []
    raw = "\0"
    for lap in range(2):                      # second lap must equal the first: the loop is periodic
        rec = []
        for t in range(PERIOD):
            tick = lap * PERIOD + t
            ev = "\0"
            if tick % POLL == 0:
                raw = KEY if contact[t] else "\0"
                ev = kp.scan(raw, tick)
            rec.append(dict(contact=contact[t], raw=raw, state=kp.state, ev=ev, enter=kp.state_enter_tick,
                            cand=kp.candidate_key, dt=(tick - kp.state_enter_tick) if kp.state == DEBOUNCE else None,
                            down=PRESS_AT <= t < RELEASE_AT))
        if lap == 0:
            first = [(r["state"], r["ev"]) for r in rec]
    assert first == [(r["state"], r["ev"]) for r in rec], "not periodic"
    events = [t for t, r in enumerate(rec) if r["ev"] != "\0"]
    assert len(events) == 1, events
    st = [r["state"] for r in rec]
    ev_t = events[0]
    dropped = any(st[t] == DEBOUNCE and st[t + 1] == IDLE for t in range(ev_t))
    assert dropped, "no bounce dropout before the press"
    return rec, ev_t


# ---------------------------------------------------------------- layout
KP_C0 = 2                       # keypad: caps 5 wide, 1 gap
TL_C0, TL_W = 34, 76            # timeline columns
LABEL_C = 26
PANE_C0, PANE_C1 = 113, COLS - 3
R_RULER, R_TICK = 1, 2
R_CON, R_PIN, R_FSM, R_EVT = 4, 7, 11, 13       # top row of each signal (waves use two rows)


def glow(age):
    """Phosphor tone for a column `age` ms behind now."""
    return "t5" if age < 4 else "t4" if age < 18 else "t3" if age < 45 else "t2"


def wave(g, r, col, level, prev, cls):
    """Two-row square wave: high level on row r, low level on row r+1, edges as |."""
    if level != prev:
        g.put(col, r, "_", cls)
        g.put(col, r + 1, "|", "t5" if cls in ("t5", "t4") else cls)
    elif level:
        g.put(col, r, "_", cls)
    else:
        g.put(col, r + 1, "_", cls)


def draw_keypad(g, scan):
    """Idle keypad with column `scan` highlighted (4-frame sprite)."""
    for c in range(4):
        x = KP_C0 + c * 6
        g.ctext(x + 2, 1, f"c{c}", "t4" if c == scan else "t1")
    for r in range(4):
        for c in range(4):
            x, y = KP_C0 + c * 6, 3 + r * 3
            cap = "t3" if c == scan else "t1"
            g.text(x, y, ".---.", cap)
            g.text(x, y + 2, "'---'", cap)
            g.put(x, y + 1, "|", cap)
            g.put(x + 4, y + 1, "|", cap)
            g.put(x + 2, y + 1, KEY_MAP[r][c], "t4" if c == scan else "t2")


def draw_pressed(g, f, rec):
    """The physically pressed key in the accent class, over the idle keypad."""
    if not rec[f]["down"]:
        return
    x, y = KP_C0 + KEY_C * 6, 3 + KEY_R * 3
    g.cover(x, y, x + 4, y + 2)                         # hide the raised cap
    g.text(x, y + 1, ".---.", "a")                      # the cap sunk by a row, flush with the board
    g.text(x, y + 2, "|_" + KEY + "_|", "a")
    g.cover(x, y, x + 4, y + 2)


def draw_timeline(g, f, rec, ev_t):
    left = TL_C0
    # labels
    for r, s in ((R_CON + 1, "contact"), (R_PIN + 1, "row pin"), (R_FSM, "fsm"), (R_EVT + 1, "event")):
        g.text(LABEL_C, r, s, "t2")
    g.rtext(TL_C0 + TL_W - 1, 0, "now", "t2")
    g.rtext(TL_C0 + TL_W - 5, 0, "", "t2")

    ts = [(f - (TL_W - 1 - j)) % PERIOD for j in range(TL_W)]
    for j, t in enumerate(ts):
        col = left + j
        tp = (t - 1) % PERIOD
        # ruler: a tick per ms, taller every 10 ms
        if t % 10 == 0:
            g.put(col, R_TICK, "|", "t3")
            lab = f"{t:02d}"
            if j >= 0 and j + len(lab) <= TL_W - 6 and j > 0:
                g.text(col - 1, R_RULER, lab, "t3")
        elif t % 5 == 0:
            g.put(col, R_TICK, "'", "t2")
        # contact and pin, in phosphor tones that fade with age
        age = TL_W - 1 - j
        wave(g, R_CON, col, rec[t]["contact"], rec[tp]["contact"], glow(age))
        pin, pin_p = int(rec[t]["raw"] != "\0"), int(rec[tp]["raw"] != "\0")
        wave(g, R_PIN, col, pin, pin_p, glow(age))
        # event pulse
        g.put(col, R_EVT + 1, "_", "t1")
        if rec[t]["ev"] != "\0":
            age = (f - t) % PERIOD
            cls = "a" if age < 24 else ("b" if age < 48 else "t2")
            g.put(col, R_EVT + 1, "|", cls)
            g.put(col, R_EVT, rec[t]["ev"], cls)

    # fsm band: runs of equal state
    j = 0
    while j < TL_W:
        s = rec[ts[j]]["state"]
        k = j
        while k + 1 < TL_W and rec[ts[k + 1]]["state"] == s:
            k += 1
        true_start = j > 0 and True
        name = NAMES[s]
        cls = ("t2", "t3", "t4")[s]
        if s == IDLE and k - j < 4:
            for c in range(j, k + 1):
                g.put(left + c, R_FSM, ".", "t1")
        if j > 0:
            g.put(left + j, R_FSM, "|", "t3")
        p = j + 1 if j > 0 else j
        if p + len(name) <= k + 1:
            g.text(left + p, R_FSM, name, cls)
        j = k + 1


def draw_pane(g, f, rec):
    now = rec[f]
    x0, x1 = PANE_C0, PANE_C1
    scan = f % 4
    g.box(x0 - 2, 2, x1 + 2, 14, "t1")
    rows = [
        ("SysTick_Get_Ms()", f"{f:4d}", "t4"),
        (f"cols[{scan}]", "SET", "t3"),
        ("raw_scan()", "'" + KEY + "'" if now["raw"] != "\0" else "'\\0'", "a" if now["raw"] != "\0" else "t3"),
        ("kp->state", NAMES[now["state"]], ("t3", "t4", "t4")[now["state"]]),
        ("debounce", f"{now['dt']:2d}/{DEBOUNCE_MS} ms" if now["dt"] is not None else "-", "t3"),
    ]
    for i, (k, v, cls) in enumerate(rows):
        r = 4 + 2 * i
        g.text(x0, r, k, "t2")
        g.rtext(x1, r, v, cls)


def build():
    rec, ev_t = simulate()
    print(f"press at {PRESS_AT} ms, event '{KEY}' at {ev_t} ms, release at {RELEASE_AT} ms")
    frames, pressed = [], []
    for f in range(PERIOD):
        p = Grid(COLS, ROWS)
        draw_pressed(p, f, rec)
        pressed.append(p)
        g = Grid(COLS, ROWS)
        draw_timeline(g, f, rec, ev_t)
        draw_pane(g, f, rec)
        frames.append(g)
    m = Movie(COLS, ROWS, title="STM32F411 bare-metal keypad driver",
              desc="A logic-analyser view of one key press on a 4x4 keypad driven by a bare-metal STM32F411 driver. "
                   "The scan column steps across the keypad while the switch contact chatters; the debounce state "
                   "machine falls back to IDLE when a bounce drops out, then reaches PRESSED once the contact has "
                   "been stable for 20 ms and Keypad_Scan returns the key exactly once, even though it stays held.")
    kp = []
    for c in range(4):
        k = Grid(COLS, ROWS)
        draw_keypad(k, c)
        kp.append(k)
    m.add(kp, FPS)
    m.add(frames, FPS)
    m.add(pressed, FPS, opaque=True)
    return m


if __name__ == "__main__":
    m = build()
    if "--preview" in sys.argv:
        m.preview("stm32.png", times=(0.5, 2.0, 3.5, 4.5, 6.0, 7.5, 9.0))
        m.preview("stm32-light.png", times=(4.2, 7.0), theme="light")
        m.preview("stm32-seam.png", times=(9.9, 0.0))
    m.write("stm32.svg")
