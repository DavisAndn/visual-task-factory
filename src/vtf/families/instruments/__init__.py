"""Instruments family: pressure gauges, syringes and analog clocks, read to the tick.

Clean renders of these are easy for current frontier models; they matter because people photograph
real instruments and ask AI to read them (see `vtf photos`), and a misread syringe is a dosing error.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from PIL import Image

from ...core.schema import GateResult, LayoutManifest
from ..drawing import probe_check
from .questions import TEMPLATES
from .render import render
from .scene import DEFAULTS, InstrumentScene, decimals_of, reading, sample_scene


class InstrumentsFamily:
    name = "instruments"
    defaults = DEFAULTS
    templates = TEMPLATES
    questions_per_scene = 2
    # Two gauges (or clocks) differ only by the needle (hands): image hashes of different tasks sit
    # closer than for charts, so only near-identical images count as copies.
    validation = {"dedup_max_hamming": 3}

    def sample_scene(self, rng: random.Random, cfg: dict[str, Any]) -> InstrumentScene:
        return sample_scene(rng, cfg)

    def scene_from_dict(self, data: dict[str, Any]) -> InstrumentScene:
        return InstrumentScene.model_validate(data)

    def render(self, scene: InstrumentScene) -> tuple[Image.Image, LayoutManifest]:
        return render(scene)

    def check_render(self, scene: InstrumentScene, image: Image.Image, layout: LayoutManifest) -> GateResult:
        return probe_check(image, layout)

    def perturb(
        self, scene: InstrumentScene, layout: LayoutManifest, params: dict[str, Any]
    ) -> Iterator[tuple[str, float, InstrumentScene]]:
        """Move the needle, the dose or the time by a few ticks (minutes), smallest move first."""
        if scene.kind == "clock":
            now = scene.hour % 12 * 60 + scene.minute
            for k in (1, 2, 3, 5, 10, 15):
                for sign in (1, -1):
                    later = (now + sign * k) % 720
                    if "until" in params:  # keep "minutes until" sensible: still ahead, under two hours
                        h, m = (int(x) for x in params["until"].split(":"))
                        if not 5 <= (h % 12 * 60 + m - later) % 720 <= 120:
                            continue
                    edited = scene.model_copy(update={"hour": later // 60 or 12, "minute": later % 60})
                    yield f"time {scene.hour}:{scene.minute:02d} → {edited.hour}:{edited.minute:02d}", k / 720, edited
            return
        scale = scene.inner or scene.scale
        assert scale is not None
        base = reading(scene)
        for k in (1, 2, 3, 5):
            for sign in (1, -1):
                new = round(base + sign * k * scale.minor, decimals_of(scale.minor))
                if not 0.05 * scale.top <= new <= 0.95 * scale.top:
                    continue
                value = round(new * scene.factor, 6) if scene.inner is not None else new
                if scene.redline is not None and value >= scene.redline:
                    continue
                if "target" in params and new > params["target"] - scale.minor + 1e-9:
                    continue  # the top-up must stay a positive amount
                edited = scene.model_copy(update={"value": value})
                what = "needle" if scene.kind == "gauge" else "dose"
                yield f"{what}: {base:g} → {new:g} {scale.unit}", k * scale.minor / scale.top, edited

    def facets(self, scene: dict[str, Any]) -> dict[str, str]:
        kind, level = scene["kind"], scene["level"]
        facets = {"instrument": kind, "level": f"L{level}", "difficulty": f"{kind} L{level}"}
        if kind == "gauge":
            facets["scale"] = "two scales" if scene.get("inner") else "one scale"
        elif kind == "syringe":
            facets["scale"] = f"{scene['scale']['top']:g} {scene['scale']['unit']}"
        else:
            facets["scale"] = f"{scene['numerals']} numerals" if scene["numerals"] != "none" else "ticks only"
        return facets
