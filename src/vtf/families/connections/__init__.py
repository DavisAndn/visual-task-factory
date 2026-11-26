"""Connections family: patch-panel cables, valve networks and gear trains, where the answer depends on
following connections. Their difficulty dial is size: more cables, pipes and gears to trace."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from PIL import Image

from ...core.schema import GateResult, LayoutManifest
from ..drawing import probe_check
from .questions import TEMPLATES
from .render import render
from .scene import DEFAULTS, ConnectionScene, sample_scene


class ConnectionsFamily:
    name = "connections"
    defaults = DEFAULTS
    templates = TEMPLATES
    questions_per_scene = 2
    # Diagrams of one kind share most of their pixels (panels, grids): only near-identical images
    # count as copies.
    validation = {"dedup_max_hamming": 3}

    def sample_scene(self, rng: random.Random, cfg: dict[str, Any]) -> ConnectionScene:
        return sample_scene(rng, cfg)

    def scene_from_dict(self, data: dict[str, Any]) -> ConnectionScene:
        return ConnectionScene.model_validate(data)

    def render(self, scene: ConnectionScene) -> tuple[Image.Image, LayoutManifest]:
        return render(scene)

    def check_render(self, scene: ConnectionScene, image: Image.Image, layout: LayoutManifest) -> GateResult:
        return probe_check(image, layout)

    def perturb(
        self, scene: ConnectionScene, layout: LayoutManifest, params: dict[str, Any]
    ) -> Iterator[tuple[str, float, ConnectionScene]]:
        """Re-plug one cable into a nearby port, or flip one valve. Gear trains have no small edit
        that keeps every other mesh as it was, so they get no contrast pairs."""
        if scene.kind == "cables":
            c = scene.cables
            n = len(c.perm)
            i = params["port"] if "port" in params else c.perm.index(params["right"])
            for step in (1, 2, 3, 4):
                for sign in (1, -1):
                    target = c.perm[i] + sign * step
                    if not 0 <= target < n:
                        continue
                    other = c.perm.index(target)
                    perm = list(c.perm)
                    perm[i], perm[other] = perm[other], perm[i]
                    edited = scene.model_copy(update={"cables": c.model_copy(update={"perm": perm})})
                    yield (f"cables from ports {i + 1} and {other + 1} swap Panel 2 ports "
                           f"{c.right[c.perm[i]]} and {c.right[c.perm[other]]}"), step / n, edited
        elif scene.kind == "valves":
            v = scene.valves
            for p in v.valves:
                closed = sorted(set(v.closed) ^ {p})
                edited = scene.model_copy(update={"valves": v.model_copy(update={"closed": closed})})
                state = "opened" if p in v.closed else "closed"
                yield f"the valve on pipe {'-'.join(v.pipes[p])} {state}", 1 / len(v.valves), edited

    def facets(self, scene: dict[str, Any]) -> dict[str, str]:
        kind, level = scene["kind"], scene["level"]
        facets = {"diagram": kind, "level": f"L{level}", "difficulty": f"{kind} L{level}"}
        if kind == "cables":
            facets["size"] = f"{len(scene['cables']['perm'])} cables" + (", colored" if scene["cables"]["colored"] else "")
        elif kind == "valves":
            facets["size"] = f"{len(scene['valves']['valves'])} valves"
        else:
            facets["size"] = f"{len(scene['gears']['gears'])} gears"
        return facets
