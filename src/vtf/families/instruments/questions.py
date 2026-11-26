"""Question templates for instruments. Answers come from the scene: the needle value, the dose,
the clock time. Readings sit on a tick, so a reading tolerance of just under half a tick accepts
the right tick and no other."""

from __future__ import annotations

import random
from typing import Any

from ...core.family import Question, Solution
from ...core.schema import Answer, LayoutManifest
from .scene import InstrumentScene, clock_minutes, decimals_of, reading


def _fmt(value: float, step: float) -> str:
    return f"{value:.{decimals_of(step)}f}"


class InstrumentTemplate:
    id = ""
    skill = ""
    weight = 1.0
    kinds: tuple[str, ...] = ()

    def applies(self, scene: InstrumentScene) -> bool:
        return scene.kind in self.kinds

    def instantiate(self, scene: InstrumentScene, layout: LayoutManifest, rng: random.Random) -> Question | None:
        raise NotImplementedError

    def solve(self, scene: InstrumentScene, params: dict[str, Any], layout: LayoutManifest) -> Solution:
        raise NotImplementedError

    def _question(self, text: str, params: dict[str, Any], depends_on: list[str]) -> Question:
        return Question(template=self.id, skill=self.skill, text=text, params=params, depends_on=depends_on)


class GaugeRead(InstrumentTemplate):
    id, skill, kinds = "gauge_read", "read a scale", ("gauge",)

    def applies(self, scene: InstrumentScene) -> bool:
        return scene.kind == "gauge" and scene.inner is None

    def instantiate(self, scene, layout, rng):
        return self._question(f"What does the gauge read, in {scene.scale.unit}?", {}, ["needle", "glabel:*", "unit"])

    def solve(self, scene, params, layout):
        sc = scene.scale
        trace = [f"Needle on the tick {_fmt(scene.value, sc.minor)} {sc.unit} (ticks every {_fmt(sc.minor, sc.minor)})"]
        return Solution(Answer(value=scene.value, type="number", tolerance=round(0.45 * sc.minor, 6), unit=sc.unit), trace)


class GaugeReadInner(InstrumentTemplate):
    id, skill, kinds = "gauge_read_inner", "pick the right scale", ("gauge",)

    def applies(self, scene: InstrumentScene) -> bool:
        return scene.kind == "gauge" and scene.inner is not None

    def instantiate(self, scene, layout, rng):
        unit = scene.inner.unit
        text = f"This gauge has two scales. What does it read on the red {unit} scale?"
        return self._question(text, {}, ["needle", "ilabel:*", "unit:inner"])

    def solve(self, scene, params, layout):
        inner = scene.inner
        value = reading(scene)
        trace = [f"Needle at {scene.value:.2f} {scene.scale.unit} = {_fmt(value, inner.minor)} {inner.unit}, "
                 f"on a tick of the inner scale (ticks every {_fmt(inner.minor, inner.minor)} {inner.unit})"]
        return Solution(Answer(value=value, type="number", tolerance=round(0.45 * inner.minor, 6), unit=inner.unit), trace)


class GaugeHeadroom(InstrumentTemplate):
    id, skill, kinds, weight = "gauge_headroom", "read and subtract", ("gauge",), 0.6

    def applies(self, scene: InstrumentScene) -> bool:
        return scene.kind == "gauge" and scene.inner is None and scene.redline is not None

    def instantiate(self, scene, layout, rng):
        text = f"How much further can the reading rise before it reaches the red zone, in {scene.scale.unit}?"
        return self._question(text, {}, ["needle", "redzone", "glabel:*", "unit"])

    def solve(self, scene, params, layout):
        sc = scene.scale
        gap = round(scene.redline - scene.value, decimals_of(sc.minor))
        trace = [f"Red zone starts at {_fmt(scene.redline, sc.minor)}", f"Needle at {_fmt(scene.value, sc.minor)}",
                 f"{_fmt(scene.redline, sc.minor)} - {_fmt(scene.value, sc.minor)} = {_fmt(gap, sc.minor)} {sc.unit}"]
        return Solution(Answer(value=gap, type="number", tolerance=round(0.45 * sc.minor, 6), unit=sc.unit), trace)


def _drawn_up(scene: InstrumentScene) -> str:
    return "insulin" if scene.scale.unit == "units" else "liquid"


class SyringeRead(InstrumentTemplate):
    id, skill, kinds = "syringe_read", "read a scale", ("syringe",)

    def instantiate(self, scene, layout, rng):
        unit = scene.scale.unit
        text = (f"How much {_drawn_up(scene)} has been drawn into this syringe, in {unit}? "
                "Read the scale at the edge of the dark rubber stopper nearest the needle.")
        return self._question(text, {}, ["stopper", "slabel:*", "unit"])

    def solve(self, scene, params, layout):
        sc = scene.scale
        trace = [f"Stopper edge on the tick {_fmt(scene.value, sc.minor)} {sc.unit} (ticks every {_fmt(sc.minor, sc.minor)})"]
        return Solution(Answer(value=scene.value, type="number", tolerance=round(0.45 * sc.minor, 6), unit=sc.unit), trace)


class SyringeTopUp(InstrumentTemplate):
    id, skill, kinds, weight = "syringe_top_up", "read and subtract", ("syringe",), 0.6

    def applies(self, scene: InstrumentScene) -> bool:
        return scene.kind == "syringe" and self._targets(scene) != []

    @staticmethod
    def _targets(scene: InstrumentScene) -> list[float]:
        """Prescribed doses: labelled values at least three ticks above the current dose (closer, and
        a reading slip of half a tick is more than the 20% the ambiguity gate allows)."""
        sc = scene.scale
        return [round(k * sc.major, 6) for k in range(1, round(sc.top / sc.major) + 1)
                if k * sc.major >= scene.value + 3 * sc.minor - 1e-9]

    def instantiate(self, scene, layout, rng):
        sc = scene.scale
        target = rng.choice(self._targets(scene))
        text = (f"The prescribed dose is {sc.label(target)} {sc.unit}. How much more must be drawn up, in {sc.unit}? "
                "Read the syringe at the edge of the dark rubber stopper nearest the needle.")
        return self._question(text, {"target": target}, ["stopper", "slabel:*", "unit"])

    def solve(self, scene, params, layout):
        sc = scene.scale
        more = round(params["target"] - scene.value, decimals_of(sc.minor))
        trace = [f"Drawn up: {_fmt(scene.value, sc.minor)} {sc.unit}",
                 f"{sc.label(params['target'])} - {_fmt(scene.value, sc.minor)} = {_fmt(more, sc.minor)} {sc.unit}"]
        return Solution(Answer(value=more, type="number", tolerance=round(0.45 * sc.minor, 6), unit=sc.unit), trace)


def _hand_refs(scene: InstrumentScene) -> list[str]:
    return ["hand:hour", "hand:minute"] + (["numeral:*"] if scene.numerals != "none" else [])


class ClockTime(InstrumentTemplate):
    id, skill, kinds = "clock_time", "read a clock", ("clock",)

    def instantiate(self, scene, layout, rng):
        return self._question("What time does the clock show?", {}, _hand_refs(scene))

    def solve(self, scene, params, layout):
        trace = [f"Hour hand just past {scene.hour}, minute hand on {scene.minute} minutes"]
        return Solution(Answer(value=f"{scene.hour}:{scene.minute:02d}", type="time", tolerance=1), trace)


class ClockUntil(InstrumentTemplate):
    id, skill, kinds, weight = "clock_until", "read and count on", ("clock",), 0.6

    def instantiate(self, scene, layout, rng):
        wait = rng.choice([d for d in range(12, 96) if d % 5])
        later = (clock_minutes(scene) + wait) % 720
        hour, minute = later // 60 or 12, later % 60
        text = f"How many minutes are left until {hour}:{minute:02d}?"
        return self._question(text, {"until": f"{hour}:{minute:02d}"}, _hand_refs(scene))

    def solve(self, scene, params, layout):
        hour, minute = (int(part) for part in params["until"].split(":"))
        wait = (hour % 12 * 60 + minute - clock_minutes(scene)) % 720
        trace = [f"Clock shows {scene.hour}:{scene.minute:02d}", f"{params['until']} is {wait} minutes later"]
        return Solution(Answer(value=wait, type="integer", tolerance=1), trace)


TEMPLATES = [GaugeRead(), GaugeReadInner(), GaugeHeadroom(), SyringeRead(), SyringeTopUp(), ClockTime(), ClockUntil()]
