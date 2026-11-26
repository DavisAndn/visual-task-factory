"""Instrument scenes: a pressure gauge, a syringe or an analog clock, with its exact reading.

Readings always sit exactly on a scale tick (a whole minute, for clocks), so the reference answer
is what a careful person reads, not an estimate between marks. Difficulty levels, 1 to 4:

  gauge    1 two ticks per labelled step · 2 four or five · 3 psi and bar scales, read the inner
           bar scale · 4 the same with finer bar ticks
  syringe  1 20 or 60 mL, 1-2 mL ticks · 2 5 or 10 mL, 0.2 mL ticks · 3 3 mL at 0.1 mL, or a
           50-unit insulin syringe · 4 1 mL or 0.5 mL tuberculin syringes, 0.01 mL ticks
  clock    1 Arabic numerals · 2 Roman numerals · 3 ticks only · 4 ticks only, a second hand, and
           hour and minute hands of similar length
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

from pydantic import BaseModel

from ..drawing import pick

PSI_PER_BAR = 14.5038

DEFAULTS: dict[str, Any] = {
    "kinds": {"gauge": 1.0, "syringe": 1.0, "clock": 1.0},
    "levels": [1.0, 1.0, 1.0, 1.0],  # weight of difficulty levels 1-4 (see this module's docstring)
    "redline": 0.5,  # probability a single-scale gauge has a red zone (levels 2-4)
    "dpi": 110,
}


class Scale(BaseModel):
    """A linear scale from 0 to `top`: a label every `major`, a tick every `minor`."""

    unit: str
    top: float
    major: float
    minor: float
    decimals: int = 0  # decimals in the printed labels

    @property
    def steps(self) -> int:
        return round(self.top / self.minor)

    @property
    def per_label(self) -> int:
        return round(self.major / self.minor)

    def label(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"


class InstrumentScene(BaseModel):
    kind: Literal["gauge", "syringe", "clock"]
    level: int
    dpi: int = 110
    title: str = ""
    # Gauge: outer scale, optional inner scale in another unit (primary units per inner unit), the
    # needle in primary units, and where a red zone starts. Syringe: its scale and the dose.
    scale: Scale | None = None
    inner: Scale | None = None
    factor: float = 1.0
    redline: float | None = None
    value: float = 0.0
    # Clock.
    hour: int = 12
    minute: int = 0
    seconds: int | None = None  # a second hand is drawn when set
    numerals: Literal["arabic", "roman", "none"] = "arabic"
    close_hands: bool = False


GAUGE_TITLES = ["PRESSURE", "HYDRAULIC", "STEAM", "AIR", "WATER", "OIL"]
# (top, major, minor) for one-scale gauges, by level.
GAUGE_SINGLE = {
    1: [(10, 1, 0.5), (16, 2, 1), (60, 10, 5), (100, 10, 5), (160, 20, 10), (400, 50, 25)],
    2: [(10, 1, 0.2), (16, 2, 0.5), (25, 5, 1), (40, 5, 1), (60, 10, 2), (100, 10, 2), (160, 20, 5),
        (250, 50, 10), (400, 50, 10), (600, 100, 20)],
}
# Dual gauges: outer psi scale (top, major, minor), inner bar scale (major, minor), by level.
GAUGE_DUAL = {
    3: [(60, 10, 2, 1, 0.5), (100, 10, 2, 1, 0.5), (160, 20, 5, 2, 1), (300, 50, 10, 5, 1)],
    4: [(60, 10, 2, 0.5, 0.1), (100, 10, 2, 1, 0.2), (160, 20, 5, 2, 0.5), (300, 50, 10, 5, 1)],
}
# (capacity, minor, label step, unit, label decimals), by level.
SYRINGES = {
    1: [(20, 1, 5, "mL", 0), (60, 2, 10, "mL", 0)],
    2: [(10, 0.2, 1, "mL", 0), (5, 0.2, 1, "mL", 0)],
    3: [(3, 0.1, 0.5, "mL", 1), (50, 1, 5, "units", 0)],
    4: [(1, 0.01, 0.1, "mL", 1), (0.5, 0.01, 0.1, "mL", 1)],
}
NUMERALS = {1: "arabic", 2: "roman", 3: "none", 4: "none"}


def decimals_of(step: float) -> int:
    text = f"{step:.6f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def on_tick(rng: random.Random, steps: int, per_label: int, low: float = 0.12, high: float = 0.92) -> int:
    """A tick index between `low` and `high` of the scale that is not a labelled tick."""
    choices = [k for k in range(math.ceil(low * steps), math.floor(high * steps) + 1) if k % per_label]
    return rng.choice(choices)


def sample_scene(rng: random.Random, cfg: dict[str, Any]) -> InstrumentScene:
    kind = pick(rng, cfg["kinds"])
    level = pick(rng, cfg["levels"])
    dpi = int(cfg.get("dpi", 110))
    if kind == "gauge":
        return _gauge(rng, level, dpi, float(cfg.get("redline", 0.5)))
    if kind == "syringe":
        cap, minor, step, unit, decimals = rng.choice(SYRINGES[level])
        scale = Scale(unit=unit, top=cap, major=step, minor=minor, decimals=decimals)
        k = on_tick(rng, scale.steps, scale.per_label, 0.08)
        return InstrumentScene(kind="syringe", level=level, dpi=dpi, scale=scale,
                               value=round(k * minor, decimals_of(minor)))
    hour, minute = rng.randint(1, 12), rng.randint(0, 59)
    if (hour, minute) in ((10, 10), (1, 50)):  # the poses clocks are photographed in: avoid
        minute = (minute + 17) % 60
    return InstrumentScene(kind="clock", level=level, dpi=dpi, hour=hour, minute=minute,
                           seconds=rng.randint(5, 55) if level == 4 else None,
                           numerals=NUMERALS[level], close_hands=level == 4)


def _gauge(rng: random.Random, level: int, dpi: int, redline_p: float) -> InstrumentScene:
    title = rng.choice(GAUGE_TITLES)
    if level <= 2:
        top, major, minor = rng.choice(GAUGE_SINGLE[level])
        unit = "bar" if top <= 40 else "psi" if top <= 160 else "kPa"
        scale = Scale(unit=unit, top=top, major=major, minor=minor, decimals=decimals_of(major))
        k = on_tick(rng, scale.steps, scale.per_label, 0.15)
        value = round(k * minor, decimals_of(minor))
        redline = None
        if level == 2 and rng.random() < redline_p:
            majors = [m * major for m in range(1, round(top / major)) if m * major >= value + major]
            redline = rng.choice(majors) if majors else None
        return InstrumentScene(kind="gauge", level=level, dpi=dpi, title=title, scale=scale, value=value, redline=redline)
    top, major, minor, bmajor, bminor = rng.choice(GAUGE_DUAL[level])
    scale = Scale(unit="psi", top=top, major=major, minor=minor, decimals=decimals_of(major))
    btop = math.floor(top / PSI_PER_BAR / bminor + 1e-9) * bminor
    inner = Scale(unit="bar", top=round(btop, 6), major=bmajor, minor=bminor, decimals=decimals_of(bmajor))
    k = on_tick(rng, inner.steps, inner.per_label, 0.12)
    bar = round(k * bminor, decimals_of(bminor))
    return InstrumentScene(kind="gauge", level=level, dpi=dpi, title=title, scale=scale, inner=inner,
                           factor=PSI_PER_BAR, value=round(bar * PSI_PER_BAR, 6))


def reading(scene: InstrumentScene) -> float:
    """What the instrument reads on the scale a question asks about (the inner one, if any)."""
    if scene.inner is not None:
        return round(scene.value / scene.factor, decimals_of(scene.inner.minor))
    return scene.value


def clock_minutes(scene: InstrumentScene) -> int:
    return scene.hour % 12 * 60 + scene.minute
