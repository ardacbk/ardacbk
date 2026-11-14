"""Raspberry Pi remote control: four processes talking, drawn as they talk.

The protocol is simulated for real: ClientNode sends a command, ServerNode's
router applies the prefix rule (sensor/relay/key -> DigitalIONode, temp/humidity
-> EnvSensorNode), the node handles it (the relay flips, the EnvSensorNode
writes 't' to the UART and its virtual sensor thread answers 25.3...) and the
reply travels back. The simulation emits a timeline of flights (literal message
text moving along a wire) and state changes; the frames are drawn from it.
"""
import sys

from engine import STRIP, Grid, Movie

COLS, ROWS = STRIP
FPS = 10
SPEED = 7                       # message columns per frame

# ---------------------------------------------------------------- the processes


class VirtualEnvSensor:
    """The second thread of EnvSensorNode: answers a character on the UART."""
    ANSWERS = {"t": "25.3", "h": "60.0"}

    def respond(self, ch):
        return self.ANSWERS[ch]


class EnvSensorNode:
    PORT = 7003

    def __init__(self):
        self.thread = VirtualEnvSensor()

    def handle(self, msg):
        kind = msg.rstrip(":")
        ch = {"temp": "t", "humidity": "h"}[kind]
        answer = self.thread.respond(ch)                  # tx ch, rx answer on /dev/serial0
        return f"{kind} {answer}:", [("uart", ch, answer)]


class DigitalIONode:
    PORT = 7002

    def __init__(self):
        self.relay, self.key = 0, "7"

    def handle(self, msg):
        if msg.startswith("relay") and msg != "relayDurum:":
            self.relay = int(msg.split()[1].rstrip(":"))
            return "relay ok:", [("relay", self.relay)]
        if msg == "relayDurum:":
            return f"relay {self.relay}:", []
        if msg == "key:":
            return f"key {self.key}:", [("key", self.key)]
        if msg == "sensorDurum:":
            return "sensorDurum 1:", []
        raise ValueError(msg)


class ServerNode:
    PORT = 7001

    def __init__(self, dio, env):
        self.rules = [(("sensor", "relay", "key"), dio), (("temp", "humidity"), env)]

    def route(self, msg):
        """The prefix rule: returns (rule index, node)."""
        for i, (prefixes, node) in enumerate(self.rules):
            if msg.startswith(prefixes):
                return i, node
        raise ValueError(msg)


# ---------------------------------------------------------------- geometry
CLIENT = (2, 2, 30, 14)
SERVER = (54, 2, 73, 14)
DIO = (86, 2, 135, 8)
ENV = (86, 9, 135, 14)
PI = (50, 0, 138, 15)
LANE_C = (8, 31, 53)             # row, first and last wire column
LANE_DIO = (5, 74, 85)
LANE_ENV = (12, 74, 85)
UART_TX = (11, 112, 123)
UART_RX = (12, 112, 123)
LANES = {0: LANE_DIO, 1: LANE_ENV}

# ---------------------------------------------------------------- timeline


class Timeline:
    def __init__(self):
        self.t = 0
        self.flights = []        # (f0, f1, text, lane, direction, cls, speed)
        self.sets = []           # (frame, key, value)

    def wait(self, n):
        self.t += n

    def set(self, key, value):
        self.sets.append((self.t, key, value))

    def fly(self, text, lane, direction, cls="t5", speed=SPEED):
        row, lo, hi = lane
        span = hi - lo + 1 + len(text) + 1
        n = -(-span // speed)
        self.flights.append((self.t, self.t + n, text, lane, direction, cls, speed))
        self.t += n

    def state(self, f):
        s = {}
        for fr, k, v in self.sets:
            if fr <= f:
                s[k] = v
        return s


def simulate(prev):
    dio, env = DigitalIONode(), EnvSensorNode()
    server = ServerNode(dio, env)
    tl = Timeline()
    tl.set("l1", prev[0])
    tl.set("l2", prev[1])
    tl.set("relay", 0)
    tl.wait(3)
    last = prev
    for cmd in ["relay 1:", "temp:", "key:", "relay 0:"]:
        # the user types
        tl.set("key", " ")
        tl.set("l2", "")
        tl.set("l1", "> " + cmd[:len(cmd) // 2])
        tl.wait(1)
        tl.set("l1", "> " + cmd)
        tl.wait(2)
        # client -> server
        tl.fly(cmd, LANE_C, +1)
        rule, node = server.route(cmd)
        tl.set("srv", rule)
        tl.wait(2)
        # server -> node
        lane = LANES[rule]
        tl.fly(cmd, lane, +1)
        tl.set("active", rule)
        reply, effects = node.handle(cmd)
        tl.wait(1)
        for eff in effects:
            if eff[0] == "relay":
                tl.set("relay", eff[1])
                tl.set("flip", tl.t)
                tl.wait(2)
            elif eff[0] == "key":
                tl.set("key", eff[1])
                tl.wait(3)
            elif eff[0] == "uart":
                _, ch, answer = eff
                tl.fly(ch, UART_TX, +1, "a", speed=4)
                tl.set("thr", 1)
                tl.wait(2)
                tl.fly(answer, UART_RX, -1, "a", speed=4)
                tl.set("thr", 0)
                tl.wait(1)
        # reply: node -> server -> client
        tl.fly(reply, lane, -1)
        tl.set("active", None)
        tl.wait(1)
        tl.fly(reply, LANE_C, -1)
        tl.set("srv", None)
        tl.set("l2", reply)
        last = ("> " + cmd, reply)
        tl.wait(4)
    return tl, last


# ---------------------------------------------------------------- drawing
def hline(g, lane, cls="t1"):
    row, lo, hi = lane
    for c in range(lo, hi + 1):
        g.put(c, row, "-", cls)


def draw_flight(g, fl, f):
    f0, f1, text, (row, lo, hi), d, cls, speed = fl
    k = f - f0
    L = len(text)
    if d > 0:
        head = lo - 1 + speed * k
        cells = [(head - L + 1 + i, ch) for i, ch in enumerate(text)]
    else:
        head = hi + 1 - speed * k
        cells = [(head + i, ch) for i, ch in enumerate(text)]
    xs = [c for c, _ in cells]
    for c in (min(xs) - 1, max(xs) + 1):                  # a little air around the text
        if lo <= c <= hi:
            g.put(c, row, " ", "t1")
    for c, ch in cells:
        if lo <= c <= hi:
            g.put(c, row, ch, cls)


def build():
    _, last = simulate(("", ""))
    tl, last2 = simulate(last)
    N = tl.t
    frames = []
    flips = [v for fr, k, v in tl.sets if k == "flip"]
    for f in range(N):
        s = tl.state(f)
        g = Grid(COLS, ROWS)
        # Pi outline with label
        g.box(*PI, "t1", corners="....")
        g.text(53, 0, " raspberry pi 3 ", "t2")
        g.text(2, 0, "pc", "t2")
        # boxes
        for b in (CLIENT, SERVER, DIO, ENV):
            g.box(*b, "t2")
        # lanes
        for lane in (LANE_C, LANE_DIO, LANE_ENV):
            hline(g, lane)
        for c, r in ((30, 8), (54, 8), (73, 5), (73, 12), (86, 5), (86, 12)):
            g.put(c, r, "+", "t2")
        g.ctext(40, 7, "tcp 7001", "t2")
        # client
        g.text(4, 3, "ClientNode", "t3")
        g.text(4, 4, "text menu", "t2")
        g.text(4, 11, s.get("l1", ""), "t4")
        g.text(4, 12, s.get("l2", ""), "t3")
        g.text(4, 10, "-" * 25, "t1")
        # server
        act = s.get("active")
        srv = s.get("srv")
        g.text(56, 3, "ServerNode", "t3")
        g.text(56, 4, "router", "t2")
        for i, (r0, rule, dest) in enumerate(((5, "sensor relay key", "-> tcp 7002"),
                                              (12, "temp humidity", "-> tcp 7003"))):
            cls = "t4" if srv == i else "t2"
            g.rtext(71, r0, rule, cls)
            g.rtext(71, r0 + 1, dest, cls)
        # DigitalIONode
        g.text(88, 3, "DigitalIONode", "t4" if act == 0 else "t3")
        g.text(88, 4, "tcp 7002", "t2")
        g.text(88, 6, "libgpiod", "t2")
        g.text(110, 3, "gpio17 sensor", "t2")
        relay = s.get("relay", 0)
        hot = any(0 <= f - fr < 8 for fr in flips)
        g.text(110, 4, "gpio27 relay", "t2")
        if relay:
            g.text(123, 4, "(*)", "a" if hot else "t4")
        else:
            g.text(123, 4, "( )", "a" if hot else "t2")
        key = s.get("key", " ")
        g.text(110, 5, "keypad", "t2")
        g.text(123, 5, f"[{key}]", "t5" if key != " " else "t2")
        g.text(110, 6, "4x4 matrix", "t1")
        # EnvSensorNode
        g.text(88, 10, "EnvSensorNode", "t4" if act == 1 else "t3")
        g.text(88, 11, "tcp 7003", "t2")
        g.text(104, 10, "uart /dev/serial0", "t2")
        g.text(104, 11, "main", "t2")
        g.text(109, 11, "tx", "t1")
        g.text(109, 12, "rx", "t1")
        g.text(126, 11, "thread", "t4" if s.get("thr") else "t2")
        g.text(126, 12, "virtual", "t1")
        g.text(104, 13, "t -> 25.3", "t1")
        g.text(116, 13, "h -> 60.0", "t1")
        hline(g, UART_TX, "t1")
        hline(g, UART_RX, "t1")
        g.put(124, 11, ">", "t1")
        g.put(111, 12, "<", "t1")
        # flights
        for fl in tl.flights:
            if fl[0] <= f < fl[1]:
                draw_flight(g, fl, f)
        frames.append(g)
    print(f"loop {N} frames = {N / FPS:.1f} s")
    m = Movie(COLS, ROWS, pad=(12, 14), title="Remote control system on a Raspberry Pi 3",
              desc="Four C++ processes as boxes: ClientNode on the PC, ServerNode as a router on the Pi, "
                   "DigitalIONode for GPIO and EnvSensorNode with its UART loop. Commands such as relay 1:, "
                   "temp:, key: and relay 0: travel along the wires as text, get routed by prefix, and the "
                   "replies travel back.")
    m.add(frames, FPS)
    return m, N


if __name__ == "__main__":
    m, N = build()
    if "--preview" in sys.argv:
        T = N / FPS
        m.preview("raspi.png", times=tuple(round(T * x, 2) for x in (0.07, 0.17, 0.27, 0.40, 0.52, 0.64, 0.78, 0.92)))
        m.preview("raspi-light.png", times=(T * 0.17, T * 0.52), theme="light")
    m.write("raspi.svg")
