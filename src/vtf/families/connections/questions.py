"""Question templates for connection diagrams. Every answer is computed by following the scene's
connections (a cable's ends, open pipes from the pump, meshes from the driving gear)."""

from __future__ import annotations

import random
from typing import Any

from ...core.family import Question, Solution
from ...core.schema import Answer, LayoutManifest
from .scene import ConnectionScene


class ConnectionTemplate:
    id = ""
    skill = ""
    weight = 1.0
    kinds: tuple[str, ...] = ()

    def applies(self, scene: ConnectionScene) -> bool:
        return scene.kind in self.kinds

    def instantiate(self, scene: ConnectionScene, layout: LayoutManifest, rng: random.Random) -> Question | None:
        raise NotImplementedError

    def solve(self, scene: ConnectionScene, params: dict[str, Any], layout: LayoutManifest) -> Solution:
        raise NotImplementedError

    def _question(self, text: str, params: dict[str, Any], depends_on: list[str]) -> Question:
        return Question(template=self.id, skill=self.skill, text=text, params=params, depends_on=depends_on)


def _traced(scene: ConnectionScene, rng: random.Random) -> int:
    """A cable worth tracing: one that crosses others (at least three, above level 1)."""
    c = scene.cables
    least = 1 if scene.level == 1 else 3
    options = [i for i in range(len(c.perm)) if c.crossings(i) >= least] or list(range(len(c.perm)))
    return rng.choice(options)


class CableTrace(ConnectionTemplate):
    id, skill, kinds = "cable_trace", "follow a line", ("cables",)

    def instantiate(self, scene, layout, rng):
        i = _traced(scene, rng)
        text = f"Which port on Panel 2 is the cable from port {i + 1} on Panel 1 plugged into?"
        return self._question(text, {"port": i}, [f"left:{i}", f"cable:{i}", "right:*"])

    def solve(self, scene, params, layout):
        c, i = scene.cables, params["port"]
        end = c.right[c.perm[i]]
        trace = [f"Cable from Panel 1 port {i + 1} crosses {c.crossings(i)} other cables", f"It ends at Panel 2 port {end}"]
        return Solution(Answer(value=end, type="label"), trace)


class CableTraceBack(ConnectionTemplate):
    id, skill, kinds = "cable_trace_back", "follow a line", ("cables",)

    def instantiate(self, scene, layout, rng):
        c = scene.cables
        i = _traced(scene, rng)
        text = f"Which port on Panel 1 does the cable plugged into port {c.right[c.perm[i]]} on Panel 2 come from?"
        return self._question(text, {"right": c.perm[i]}, [f"right:{c.perm[i]}", f"cable:{i}", "left:*"])

    def solve(self, scene, params, layout):
        c = scene.cables
        i = c.perm.index(params["right"])
        trace = [f"Cable at Panel 2 port {c.right[params['right']]} crosses {c.crossings(i)} other cables",
                 f"It starts at Panel 1 port {i + 1}"]
        return Solution(Answer(value=i + 1, type="integer"), trace)


def _tank_names(scene: ConnectionScene, tanks: list[int]) -> list[str]:
    return sorted((scene.valves.tanks[k].name for k in tanks), key=lambda name: int(name[1:]))


class ValvesReach(ConnectionTemplate):
    id, skill, kinds = "valves_reach", "trace a network", ("valves",)

    def instantiate(self, scene, layout, rng):
        text = "Water enters the network from pump P1. With the valves set as shown, which tanks can the water reach?"
        return self._question(text, {}, ["pump", "tank:*", "valve:*"])

    def solve(self, scene, params, layout):
        v = scene.valves
        got = _tank_names(scene, v.reachable())
        options = _tank_names(scene, list(range(len(v.tanks))))
        trace = [f"{len(v.closed)} of {len(v.valves)} valves are closed",
                 f"Open pipes from P1 reach {', '.join(got)}; the others are cut off"]
        return Solution(Answer(value=", ".join(got) or "none", type="set", options=options), trace)


class ValvesOne(ConnectionTemplate):
    id, skill, kinds = "valves_one", "trace a network", ("valves",)

    def instantiate(self, scene, layout, rng):
        v = scene.valves
        k = rng.randrange(len(v.tanks))
        text = f"Water enters the network from pump P1. With the valves set as shown, can the water reach tank {v.tanks[k].name}?"
        return self._question(text, {"tank": k}, ["pump", f"tank:{k}", "valve:*"])

    def solve(self, scene, params, layout):
        v, k = scene.valves, params["tank"]
        yes = k in v.reachable()
        trace = [f"Tank {v.tanks[k].name} is {'reached through open pipes' if yes else 'cut off by closed valves'}"]
        return Solution(Answer(value="yes" if yes else "no", type="yes_no"), trace)


class GearsCounter(ConnectionTemplate):
    id, skill, kinds = "gears_ccw", "chain of reasoning", ("gears",)

    def instantiate(self, scene, layout, rng):
        n = len(scene.gears.gears)
        return self._question("Gear A is driven clockwise. Which gears turn counterclockwise?", {},
                              ["gear:0", "arrow"] + [f"gear:{i}" for i in range(1, n)])

    def solve(self, scene, params, layout):
        g = scene.gears
        ccw = sorted(gear.label for i, gear in enumerate(g.gears) if (g.depth(i) or 0) % 2 == 1)
        still = sorted(gear.label for i, gear in enumerate(g.gears) if g.depth(i) is None)
        trace = ["Each mesh reverses the direction: gears an odd number of meshes from A turn counterclockwise"]
        if still:
            trace.append(f"{', '.join(still)} touch no other gear and don't turn")
        options = sorted(gear.label for gear in g.gears[1:])
        return Solution(Answer(value=", ".join(ccw) or "none", type="set", options=options), trace)


class GearsSame(ConnectionTemplate):
    id, skill, kinds = "gears_same", "chain of reasoning", ("gears",)

    def instantiate(self, scene, layout, rng):
        g = scene.gears
        turning = [i for i in range(1, len(g.gears)) if g.depth(i) is not None and g.depth(i) >= 2]
        if not turning:
            return None
        i = rng.choice(turning)
        text = f"Gear A is driven clockwise. Does gear {g.gears[i].label} turn the same way as gear A?"
        return self._question(text, {"gear": i}, ["gear:0", "arrow", f"gear:{i}"])

    def solve(self, scene, params, layout):
        g, i = scene.gears, params["gear"]
        depth = g.depth(i)
        trace = [f"Gear {g.gears[i].label} is {depth} meshes from A: {'even, same way' if depth % 2 == 0 else 'odd, opposite way'}"]
        return Solution(Answer(value="yes" if depth % 2 == 0 else "no", type="yes_no"), trace)


TEMPLATES = [CableTrace(), CableTraceBack(), ValvesReach(), ValvesOne(), GearsCounter(), GearsSame()]
