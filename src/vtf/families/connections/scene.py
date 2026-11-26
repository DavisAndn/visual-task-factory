"""Connection diagrams: patch-panel cables, cooling-water valve networks, and gear trains.

Each answer follows what connects to what, so it can only come from tracing the image. Levels:

  cables  1 8 cables in different colors · 2 8 in one color · 3 16 · 4 24 · 5 32, one color
  valves  1 a 4x3 network, 3 tanks, 6 valves · 2 6x5, 6 tanks, 14 valves · 3 8x7, 10 tanks, 24 valves
  gears   1 a chain of 5 · 2 9 in a branching train, plus 2 that touch nothing · 3 14 plus 3
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel

from ..drawing import pick

DEFAULTS: dict[str, Any] = {
    "kinds": {"cables": 1.0, "valves": 1.0, "gears": 1.0},
    # Weight of each difficulty level, per kind (see this module's docstring).
    "levels": {"cables": [1.0, 1.0, 1.0, 1.0, 1.0], "valves": [1.0, 1.0, 1.0], "gears": [1.0, 1.0, 1.0]},
    "dpi": 110,
}
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I, O or Q, which read like 1 and 0


class Cables(BaseModel):
    colored: bool
    perm: list[int]  # perm[i]: the Panel 2 port of the cable from Panel 1 port i
    right: list[str]  # Panel 2 port names, top to bottom (Panel 1 ports are numbered 1..n)

    def crossings(self, i: int) -> int:
        return sum((i - j) * (self.perm[i] - self.perm[j]) < 0 for j in range(len(self.perm)))


class Tank(BaseModel):
    node: tuple[int, int]
    out: tuple[int, int]  # direction of its stub pipe, away from the grid
    name: str


class Valves(BaseModel):
    width: int
    height: int
    start: tuple[int, int]  # grid node the pump feeds
    pipes: list[tuple[str, str]]  # node ids: "i,j" on the grid, "P" the pump, "T<k>" tank k
    valves: list[int]  # pipes with a valve
    closed: list[int]  # the closed ones (a subset of valves)
    tanks: list[Tank]

    def reachable(self, closed: set[int] | None = None) -> list[int]:
        """Tanks the pump's water reaches: through plain pipes and open valves."""
        shut = set(self.closed) if closed is None else closed
        adj: dict[str, list[str]] = defaultdict(list)
        for p, (a, b) in enumerate(self.pipes):
            if p not in shut:
                adj[a].append(b)
                adj[b].append(a)
        seen, todo = {"P"}, ["P"]
        while todo:
            for nxt in adj[todo.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return [k for k in range(len(self.tanks)) if f"T{k}" in seen]


class Gear(BaseModel):
    x: float
    y: float
    teeth: int
    phase: float
    parent: int | None  # the gear it meshes with on the way from the driver (gear 0)
    label: str


class Gears(BaseModel):
    module: float
    gears: list[Gear]

    def depth(self, i: int) -> int | None:
        """Meshes between gear i and the driver; None for a gear that touches nothing."""
        steps = 0
        while i != 0:
            parent = self.gears[i].parent
            if parent is None:
                return None
            i, steps = parent, steps + 1
        return steps

    def radius(self, i: int) -> float:
        return self.module * self.gears[i].teeth / 2


class ConnectionScene(BaseModel):
    kind: Literal["cables", "valves", "gears"]
    level: int
    dpi: int = 110
    cables: Cables | None = None
    valves: Valves | None = None
    gears: Gears | None = None


def sample_scene(rng: random.Random, cfg: dict[str, Any]) -> ConnectionScene:
    kind = pick(rng, cfg["kinds"])
    level = pick(rng, cfg["levels"][kind])
    dpi = int(cfg.get("dpi", 110))
    if kind == "cables":
        return ConnectionScene(kind=kind, level=level, dpi=dpi, cables=_cables(rng, level))
    if kind == "valves":
        return ConnectionScene(kind=kind, level=level, dpi=dpi, valves=_valves(rng, level))
    return ConnectionScene(kind=kind, level=level, dpi=dpi, gears=_gears(rng, level))


def _cables(rng: random.Random, level: int) -> Cables:
    n, colored = {1: (8, True), 2: (8, False), 3: (16, False), 4: (24, False), 5: (32, False)}[level]
    right = list(LETTERS[:n]) if n <= 16 else [f"{row}{col}" for row in "ABCD" for col in range(1, 9)][:n]
    return Cables(colored=colored, perm=rng.sample(range(n), n), right=right)


def _valves(rng: random.Random, level: int) -> Valves:
    W, H, ntank, nvalve = {1: (4, 3, 3, 6), 2: (6, 5, 6, 14), 3: (8, 7, 10, 24)}[level]

    def nbrs(n: tuple[int, int]) -> list[tuple[int, int]]:
        i, j = n
        return [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < W and 0 <= j + b < H]

    def key(n: tuple[int, int]) -> str:
        return f"{n[0]},{n[1]}"

    while True:
        start = (0, rng.randrange(H))
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        seen, stack = {start}, [start]
        while stack:  # a random spanning tree of the grid, then a few extra pipes that make loops
            options = [b for b in nbrs(stack[-1]) if b not in seen]
            if not options:
                stack.pop()
                continue
            nxt = rng.choice(options)
            seen.add(nxt)
            edges.append((stack[-1], nxt))
            stack.append(nxt)
        have = {frozenset(e) for e in edges}
        for a in [(i, j) for i in range(W) for j in range(H)]:
            for b in nbrs(a):
                if frozenset((a, b)) not in have and rng.random() < 0.12:
                    have.add(frozenset((a, b)))
                    edges.append((a, b))
        spots = [((W - 1, j), (1, 0)) for j in range(H)] + [((i, H - 1), (0, 1)) for i in range(1, W - 1)] + \
                [((i, 0), (0, -1)) for i in range(1, W - 1)]
        rng.shuffle(spots)
        tanks: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used = {start}
        for node, out in spots:
            if node not in used and len(tanks) < ntank:
                tanks.append((node, out))
                used.add(node)
        pipes = [(key(a), key(b)) for a, b in edges] + [("P", key(start))] + [(key(node), f"T{k}") for k, (node, _) in enumerate(tanks)]
        candidates = [p for p in range(len(pipes)) if "P" not in pipes[p]]
        valves = sorted(rng.sample(candidates, nvalve))
        closed = [p for p in valves if rng.random() < 0.45]
        names = [f"T{i + 1}" for i in rng.sample(range(ntank), ntank)]
        scene = Valves(width=W, height=H, start=start, pipes=pipes, valves=valves, closed=closed,
                       tanks=[Tank(node=node, out=out, name=names[k]) for k, (node, out) in enumerate(tanks)])
        got = scene.reachable()
        stubs = {pipes.index((key(node), f"T{k}")): k for k, (node, _) in enumerate(tanks)}
        cut_upstream = [k for k in range(ntank) if k not in got and not any(p in closed and stubs[p] == k for p in stubs)]
        # Some tanks reached, some not, and at least one cut off by a valve further up its line,
        # so the answer can't be read off the valves next to the tanks.
        if got and cut_upstream and len(closed) >= 2:
            return scene


def _gears(rng: random.Random, level: int) -> Gears:
    n, free = {1: (5, 0), 2: (9, 2), 3: (14, 3)}[level]
    m, W, H = (1.0 if level < 3 else 0.72), 100.0, 64.0
    while True:
        z0 = rng.choice([16, 18, 20, 24])
        placed: list[dict[str, Any]] = [{"x": rng.uniform(18, 35), "y": rng.uniform(18, 44), "z": z0, "R": m * z0 / 2,
                                         "phase": rng.uniform(0, 2 * math.pi), "parent": None}]
        for _ in range(4000):
            if len(placed) == n:
                break
            p = len(placed) - 1 if level == 1 else rng.randrange(len(placed))
            parent = placed[p]
            z = rng.choice([12, 14, 16, 18, 20, 24, 28, 32])
            R, th = m * z / 2, rng.uniform(0, 2 * math.pi)
            x = parent["x"] + (parent["R"] + R) * math.cos(th)
            y = parent["y"] + (parent["R"] + R) * math.sin(th)
            if not (R + 3 <= x <= W - R - 3 and R + 3 <= y <= H - R - 3):
                continue
            # Every gear but its parent keeps a clear gap of at least four tooth heights.
            if any(math.hypot(x - g["x"], y - g["y"]) < R + g["R"] + 6 * m for q, g in enumerate(placed) if q != p):
                continue
            f = (th - parent["phase"]) / (2 * math.pi / parent["z"])
            f -= round(f)  # where the parent's teeth sit at the contact; the child puts a gap there
            placed.append({"x": x, "y": y, "z": z, "R": R, "phase": th + math.pi + (f - 0.5) * 2 * math.pi / z, "parent": p})
        if len(placed) < n:
            continue
        loose: list[dict[str, Any]] = []
        for _ in range(4000):
            if len(loose) == free:
                break
            z = rng.choice([12, 14, 16, 18, 20])
            R = m * z / 2
            x, y = rng.uniform(R + 3, W - R - 3), rng.uniform(R + 3, H - R - 3)
            gap = min(math.hypot(x - g["x"], y - g["y"]) - (R + g["R"] + 2 * m) for g in placed + loose)
            if 3 * m <= gap <= 6 * m:  # near a gear, clearly not touching it: it doesn't turn
                loose.append({"x": x, "y": y, "z": z, "R": R, "phase": rng.uniform(0, 2 * math.pi), "parent": None})
        if len(loose) == free:
            break
    every = placed + loose
    labels = ["A"] + rng.sample(list(LETTERS[1:len(every) + 2]), len(every) - 1)
    gears = [Gear(x=round(g["x"], 4), y=round(g["y"], 4), teeth=g["z"], phase=round(g["phase"], 6), parent=g["parent"],
                  label=label) for g, label in zip(every, labels)]
    return Gears(module=m, gears=gears)
