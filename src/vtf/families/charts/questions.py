"""Question templates for charts.

Each template picks its parameters in `instantiate`; `solve` then computes the answer from the
scene's data, never from pixels or a model. `depends_on` lists every part of the image that the
question and its answer rely on, so the grounding gate can confirm each one is visible.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from ...core.family import Question, Solution
from ...core.schema import Answer, LayoutManifest
from .scene import DOMAIN_BY_KEY, ChartScene, Domain

ALL_TYPES = ("bar", "hbar", "grouped_bar", "stacked_bar", "line")  # charts with a value axis
BAR_TYPES = ("bar", "hbar", "grouped_bar")
STACKED = ("stacked_bar",)
PIE = ("pie",)
ORDINALS = {2: "second", 3: "third"}


def _domain(scene: ChartScene) -> Domain:
    return DOMAIN_BY_KEY[scene.domain]


def _fmt(scene: ChartScene, value: float) -> str:
    return f"{value:.{scene.decimals}f}"


def _num(scene: ChartScene, value: float) -> int | float:
    return int(round(value)) if scene.decimals == 0 else round(value, scene.decimals)


def _a(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _series_ref(scene: ChartScene, s: int) -> str:
    """Where the series name is visible: the title for a single series, otherwise the legend."""
    return "title" if len(scene.series) == 1 else f"legend:{s}"


def _mark(scene: ChartScene, s: int, i: int) -> str:
    return f"{'point' if scene.chart_type == 'line' else 'bar'}:{s}:{i}"


def _all_cats(scene: ChartScene) -> list[str]:
    return [f"cat:{i}" for i in range(len(scene.categories))]


def _all_marks(scene: ChartScene, s: int) -> list[str]:
    return [_mark(scene, s, i) for i in range(len(scene.categories))]


def _value_refs(scene: ChartScene, s: int, indices: Iterable[int]) -> list[str]:
    """Reading a value needs its printed label, or a readable value axis when nothing is printed."""
    return [f"val:{s}:{i}" for i in indices] if scene.style.value_labels else ["vtick:*"]


def _entity_and_period(scene: ChartScene, s: int, i: int) -> tuple[str, str]:
    if scene.categories_are_periods:
        return scene.series[s].name, scene.categories[i]
    return scene.categories[i], scene.series[s].name


def _read_tolerance(scene: ChartScene, layout: LayoutManifest, value: float, series: int = 0) -> float:
    """Exact when values are printed. Otherwise 5% as in ChartQA, but never tighter than a tenth
    of the gridline step, which is about as finely as anyone can read an axis. A stacked segment
    that doesn't start at zero takes two readings (its top and its bottom), so twice that."""
    if scene.style.value_labels:
        return 0.0
    single = max(0.05 * abs(value), layout.meta["tick_step"] / 10)
    readings = 2 if scene.chart_type == "stacked_bar" and series > 0 else 1
    return round(readings * single, 6)


def _segment_margins(scene: ChartScene, margins: list[float], series: Iterable[int]) -> list[float]:
    """Unprinted stacked segments above the bottom one don't share a baseline, so they are judged
    by length rather than by position on a common axis, which people do about half as precisely
    (Cleveland and McGill). Their deciding gaps count half against the ambiguity gate."""
    if scene.chart_type == "stacked_bar" and not scene.style.value_labels and any(s > 0 for s in series):
        return [m / 2 for m in margins]
    return margins


def _balanced_threshold(thresholds: list[float], values: list[float], rng: random.Random) -> float:
    """A threshold for a 'how many are above' question, picked so the count itself is spread out:
    first a target count, evenly among the counts some threshold produces, then a threshold for it.
    Picking thresholds directly makes small counts far more common, which a model could exploit."""
    by_count: dict[int, list[float]] = {}
    for t in thresholds:
        by_count.setdefault(sum(v > t for v in values), []).append(t)
    return rng.choice(by_count[rng.choice(sorted(by_count))])


class ChartTemplate:
    id = ""
    skill = ""
    weight = 1.0
    chart_types: tuple[str, ...] = ALL_TYPES
    min_categories = 2

    def applies(self, scene: ChartScene) -> bool:
        return scene.chart_type in self.chart_types and len(scene.categories) >= self.min_categories

    def instantiate(self, scene: ChartScene, layout: LayoutManifest, rng: random.Random) -> Question | None:
        raise NotImplementedError

    def solve(self, scene: ChartScene, params: dict[str, Any], layout: LayoutManifest) -> Solution:
        raise NotImplementedError

    def _question(self, text: str, params: dict[str, Any], depends_on: list[str]) -> Question:
        unique = list(dict.fromkeys(depends_on))
        return Question(template=self.id, skill=self.skill, text=text, params=params, depends_on=unique)


class CountCategories(ChartTemplate):
    id, skill, weight = "count_categories", "structure", 0.5
    chart_types = BAR_TYPES + STACKED

    def instantiate(self, scene, layout, rng):
        d = _domain(scene)
        noun = d.period[1] if scene.categories_are_periods else d.entity[1]
        return self._question(f"How many {noun} does the chart show?", {}, _all_cats(scene))

    def solve(self, scene, params, layout):
        n = len(scene.categories)
        return Solution(Answer(value=n, type="integer"), [f"count the category labels → {n}"])


class ReadValue(ChartTemplate):
    id, skill = "read_value", "retrieval"

    def instantiate(self, scene, layout, rng):
        s, i = rng.randrange(len(scene.series)), rng.randrange(len(scene.categories))
        entity, period = _entity_and_period(scene, s, i)
        d = _domain(scene)
        text = f"What was the {d.metric} for {entity} in {period}? Answer in {d.answer_unit}."
        refs = [f"cat:{i}", _series_ref(scene, s), _mark(scene, s, i), *_value_refs(scene, s, [i])]
        return self._question(text, {"series": s, "index": i}, refs)

    def solve(self, scene, params, layout):
        s, i = params["series"], params["index"]
        value = scene.series[s].values[i]
        entity, period = _entity_and_period(scene, s, i)
        how = "read the printed value" if scene.style.value_labels else "read it against the value axis"
        answer = Answer(
            value=_num(scene, value),
            type="number",
            tolerance=_read_tolerance(scene, layout, value, s),
            unit=_domain(scene).answer_unit,
        )
        return Solution(answer, [f"find {entity} in {period}", f"{how} → {_fmt(scene, value)}"])


class Extremum(ChartTemplate):
    id, skill = "extremum", "extremum"
    min_categories = 3

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        kind = rng.choice(("highest", "lowest"))
        d, name = _domain(scene), scene.series[s].name
        if scene.categories_are_periods:
            text = f"In which {d.period[0]} did {name} have the {kind} {d.metric}?"
        else:
            text = f"Which {d.entity[0]} had the {kind} {d.metric} in {name}?"
        refs = [*_all_cats(scene), *_all_marks(scene, s), _series_ref(scene, s)]
        return self._question(text, {"series": s, "kind": kind}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        order = sorted(range(len(values)), key=values.__getitem__, reverse=params["kind"] == "highest")
        best, runner_up = order[0], order[1]
        cats = scene.categories
        trace = [
            f"compare {len(values)} values",
            f"{params['kind']}: {cats[best]} ({_fmt(scene, values[best])}), "
            f"next {cats[runner_up]} ({_fmt(scene, values[runner_up])})",
        ]
        gap = [abs(values[best] - values[runner_up])]
        return Solution(Answer(value=cats[best], type="label"), trace, margins=_segment_margins(scene, gap, [params["series"]]))


class ComparePair(ChartTemplate):
    id, skill = "compare_pair", "comparison"

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        values = scene.series[s].values
        a, b = rng.sample(range(len(values)), 2)
        if values[a] == values[b]:
            return None
        # Decide the answer first so yes and no stay balanced across the dataset.
        if (values[a] > values[b]) != (rng.random() < 0.5):
            a, b = b, a
        d, name, cats = _domain(scene), scene.series[s].name, scene.categories
        if scene.categories_are_periods:
            text = f"Was {name}'s {d.metric} higher in {cats[a]} than in {cats[b]}?"
        else:
            text = f"Did {cats[a]} have a higher {d.metric} than {cats[b]} in {name}?"
        refs = [f"cat:{a}", f"cat:{b}", _mark(scene, s, a), _mark(scene, s, b), _series_ref(scene, s)]
        return self._question(text, {"series": s, "a": a, "b": b}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        a, b = params["a"], params["b"]
        verdict = "yes" if values[a] > values[b] else "no"
        cats = scene.categories
        trace = [f"{cats[a]} = {_fmt(scene, values[a])}, {cats[b]} = {_fmt(scene, values[b])}", f"higher? → {verdict}"]
        gap = [abs(values[a] - values[b])]
        return Solution(Answer(value=verdict, type="yes_no"), trace, margins=_segment_margins(scene, gap, [params["series"]]))


class CountAbove(ChartTemplate):
    id, skill = "count_above", "counting"
    min_categories = 3

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        values = scene.series[s].values
        ticks = [t for t in layout.meta["value_ticks"] if min(values) < t < max(values)]
        if not ticks:
            return None
        threshold = _balanced_threshold(ticks, values, rng)
        d, name = _domain(scene), scene.series[s].name
        shown = d.threshold.format(f"{threshold:g}")
        if scene.categories_are_periods:
            text = f"In how many {d.period[1]} was {name}'s {d.metric} above {shown}?"
        else:
            text = f"How many {d.entity[1]} had {_a(d.metric)} {d.metric} above {shown} in {name}?"
        refs = [
            *_all_cats(scene),
            *_all_marks(scene, s),
            _series_ref(scene, s),
            *_value_refs(scene, s, range(len(values))),
        ]
        return self._question(text, {"series": s, "threshold": threshold}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        threshold = params["threshold"]
        above = [scene.categories[i] for i, v in enumerate(values) if v > threshold]
        trace = [f"above {threshold:g}: {', '.join(above) or 'none'}", f"count → {len(above)}"]
        margins = _segment_margins(scene, [abs(v - threshold) for v in values], [params["series"]])
        return Solution(Answer(value=len(above), type="integer"), trace, margins=margins)


class Difference(ChartTemplate):
    id, skill = "difference", "arithmetic"

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        a, b = rng.sample(range(len(scene.categories)), 2)
        d, name, cats = _domain(scene), scene.series[s].name, scene.categories
        if scene.categories_are_periods:
            # "Absolute": "the difference between 2019 and 2016" can fairly be answered as a negative number.
            text = (
                f"What is the absolute difference between {name}'s {d.metric} in {cats[a]} and in {cats[b]}? "
                f"Answer in {d.answer_unit}."
            )
        else:
            text = (
                f"What is the absolute difference in {d.metric} between {cats[a]} and {cats[b]} in {name}? "
                f"Answer in {d.answer_unit}."
            )
        refs = [f"cat:{a}", f"cat:{b}", _mark(scene, s, a), _mark(scene, s, b), _series_ref(scene, s)]
        refs += _value_refs(scene, s, [a, b])
        return self._question(text, {"series": s, "a": a, "b": b}, refs)

    def solve(self, scene, params, layout):
        s = params["series"]
        values = scene.series[s].values
        a, b = params["a"], params["b"]
        diff = abs(values[a] - values[b])
        tolerance = _read_tolerance(scene, layout, values[a], s) + _read_tolerance(scene, layout, values[b], s)
        answer = Answer(value=_num(scene, diff), type="number", tolerance=round(tolerance, 6), unit=_domain(scene).answer_unit)
        cats = scene.categories
        trace = [f"{cats[a]} = {_fmt(scene, values[a])}, {cats[b]} = {_fmt(scene, values[b])}", f"difference → {_fmt(scene, diff)}"]
        return Solution(answer, trace)


class RankThenRead(ChartTemplate):
    id, skill = "rank_then_read", "multi_hop"
    chart_types = BAR_TYPES
    min_categories = 4

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        rank = rng.choice((2, 3))
        d, name = _domain(scene), scene.series[s].name
        text = (
            f"What was the {d.metric} of the {d.entity[0]} with the {ORDINALS[rank]}-highest "
            f"{d.metric} in {name}? Answer in {d.answer_unit}."
        )
        refs = [
            *_all_cats(scene),
            *_all_marks(scene, s),
            _series_ref(scene, s),
            *_value_refs(scene, s, range(len(scene.categories))),
        ]
        return self._question(text, {"series": s, "rank": rank}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        rank = params["rank"]
        order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
        pick = order[rank - 1]
        value = values[pick]
        cats = scene.categories
        answer = Answer(
            value=_num(scene, value),
            type="number",
            tolerance=_read_tolerance(scene, layout, value),
            unit=_domain(scene).answer_unit,
        )
        trace = [
            "rank: " + ", ".join(cats[i] for i in order[: rank + 1]),
            f"{ORDINALS[rank]}-highest is {cats[pick]}",
            f"read value → {_fmt(scene, value)}",
        ]
        margins = [values[order[rank - 2]] - value, value - values[order[rank]]]
        return Solution(answer, trace, margins=margins)


def _leader_after(values: list[float], index: int, factor: float) -> int:
    changed = list(values)
    changed[index] *= factor
    return max(range(len(changed)), key=changed.__getitem__)


class WhatIf(ChartTemplate):
    """Change one value, then ask who leads.

    The entity named in the question wins only about 1 time in n, so guessing it from the
    question text alone does no better than picking a category at random. Changes whose
    answer can't depend on the change (halving a non-leader, doubling the leader) are skipped.
    """

    id, skill = "what_if", "counterfactual"
    chart_types = BAR_TYPES
    min_categories = 3
    VERBS = {2.0: ("doubled", "double"), 0.5: ("halved", "halve")}

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        values = scene.series[s].values
        n = len(values)
        leader = max(range(n), key=values.__getitem__)
        options = [(i, 2.0) for i in range(n) if i != leader] + [(leader, 0.5)]
        named_wins = [(i, f) for i, f in options if _leader_after(values, i, f) == i]
        other_wins = [(i, f) for i, f in options if _leader_after(values, i, f) != i]
        preferred = named_wins if rng.random() < 1 / n else other_wins
        index, factor = rng.choice(preferred or named_wins or other_wins)
        d, name, cats = _domain(scene), scene.series[s].name, scene.categories
        # The series is named again at the end: on a chart with several years, "the highest
        # visitor count" alone could mean the highest in any year (two models read it that way).
        text = (
            f"If {cats[index]}'s {d.metric} in {name} were {self.VERBS[factor][0]}, "
            f"which {d.entity[0]} would have the highest {d.metric} in {name}?"
        )
        refs = [*_all_cats(scene), *_all_marks(scene, s), _series_ref(scene, s)]
        return self._question(text, {"series": s, "index": index, "factor": factor}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        index, factor = params["index"], params["factor"]
        changed = list(values)
        changed[index] *= factor
        order = sorted(range(len(changed)), key=changed.__getitem__, reverse=True)
        best, runner_up = order[0], order[1]
        cats = scene.categories
        trace = [
            f"{self.VERBS[factor][1]} {cats[index]}: {_fmt(scene, values[index])} → {_fmt(scene, changed[index])}",
            f"highest → {cats[best]} ({_fmt(scene, changed[best])}), next {cats[runner_up]} ({_fmt(scene, changed[runner_up])})",
        ]
        margin = changed[best] - changed[runner_up]
        if factor > 1 and index in (best, runner_up):
            margin /= factor  # a doubled bar carries twice the reading error, so its lead counts half
        return Solution(Answer(value=cats[best], type="label"), trace, margins=[margin])


class TrendDecreases(ChartTemplate):
    id, skill = "trend_decreases", "trend"
    chart_types = ("line",)

    def instantiate(self, scene, layout, rng):
        s = rng.randrange(len(scene.series))
        d, name = _domain(scene), scene.series[s].name
        text = f"How many times did {name}'s {d.metric} decrease from one {d.period[0]} to the next?"
        refs = [*_all_cats(scene), *_all_marks(scene, s), _series_ref(scene, s)]
        return self._question(text, {"series": s}, refs)

    def solve(self, scene, params, layout):
        values = scene.series[params["series"]].values
        steps = [b - a for a, b in zip(values, values[1:])]
        drops = sum(1 for step in steps if step < 0)
        changes = ", ".join(("+" if step >= 0 else "−") + _fmt(scene, abs(step)) for step in steps)
        trace = [f"changes: {changes}", f"decreases → {drops}"]
        return Solution(Answer(value=drops, type="integer"), trace, margins=[abs(step) for step in steps])


class LargestIncrease(ChartTemplate):
    id, skill = "largest_increase", "multi_hop"
    chart_types = ("grouped_bar",)
    min_categories = 3

    def instantiate(self, scene, layout, rng):
        a, b = sorted(rng.sample(range(len(scene.series)), 2))
        before, after = scene.series[a], scene.series[b]
        if max(y - x for x, y in zip(before.values, after.values)) <= 0:
            return None
        d = _domain(scene)
        text = f"Which {d.entity[0]} had the largest increase in {d.metric} from {before.name} to {after.name}?"
        refs = [*_all_cats(scene), *_all_marks(scene, a), *_all_marks(scene, b), f"legend:{a}", f"legend:{b}"]
        return self._question(text, {"from": a, "to": b}, refs)

    def solve(self, scene, params, layout):
        before, after = scene.series[params["from"]].values, scene.series[params["to"]].values
        gains = [y - x for x, y in zip(before, after)]
        order = sorted(range(len(gains)), key=gains.__getitem__, reverse=True)
        best, runner_up = order[0], order[1]
        cats = scene.categories
        listed = ", ".join(f"{cats[i]} {'+' if g >= 0 else '−'}{_fmt(scene, abs(g))}" for i, g in enumerate(gains))
        trace = [f"change: {listed}", f"largest increase → {cats[best]}"]
        margins = [gains[best] - gains[runner_up], gains[best]]
        return Solution(Answer(value=cats[best], type="label"), trace, margins=margins)


class StackTotal(ChartTemplate):
    """The total of one stacked bar: the top of the stack against the axis, or its printed total."""

    id, skill = "stack_total", "aggregation"
    chart_types = STACKED

    def instantiate(self, scene, layout, rng):
        i = rng.randrange(len(scene.categories))
        d = _domain(scene)
        text = (
            f"What was the total {d.metric} of all {d.entity[1]} combined in {scene.categories[i]}? "
            f"Answer in {d.answer_unit}."
        )
        refs = [f"cat:{i}", *(f"bar:{s}:{i}" for s in range(len(scene.series)))]
        refs.append(f"tot:{i}" if scene.style.value_labels else "vtick:*")
        return self._question(text, {"index": i}, refs)

    def solve(self, scene, params, layout):
        i = params["index"]
        parts = [s.values[i] for s in scene.series]
        total = sum(parts)
        how = "read the printed total" if scene.style.value_labels else "read the top of the stack against the axis"
        answer = Answer(
            value=_num(scene, total),
            type="number",
            tolerance=_read_tolerance(scene, layout, total),
            unit=_domain(scene).answer_unit,
        )
        trace = [" + ".join(_fmt(scene, v) for v in parts) + f" = {_fmt(scene, total)}", how]
        return Solution(answer, trace)


class LargestSegment(ChartTemplate):
    """Which part of one stacked bar is biggest. Segments start at different heights, so their
    lengths can't be compared against a shared baseline."""

    id, skill = "largest_segment", "composition"
    chart_types = STACKED

    def instantiate(self, scene, layout, rng):
        i = rng.randrange(len(scene.categories))
        d = _domain(scene)
        text = f"In {scene.categories[i]}, which {d.entity[0]} had the highest {d.metric}?"
        k = len(scene.series)
        refs = [f"cat:{i}", *(f"bar:{s}:{i}" for s in range(k)), *(f"legend:{s}" for s in range(k))]
        return self._question(text, {"index": i}, refs)

    def solve(self, scene, params, layout):
        i = params["index"]
        parts = [(s.values[i], s.name) for s in scene.series]
        (best, name), (second, _) = sorted(parts, reverse=True)[:2]
        trace = [", ".join(f"{n} {_fmt(scene, v)}" for v, n in parts), f"largest → {name}"]
        margins = _segment_margins(scene, [best - second], range(len(scene.series)))
        return Solution(Answer(value=name, type="label"), trace, margins=margins)


def _pie_tolerance(scene: ChartScene) -> float:
    """Printed percentages are exact. Without them a share is judged from the slice's angle,
    which people manage to within a few percentage points."""
    return 0.0 if scene.style.value_labels else 3.0


def _pie_refs(scene: ChartScene, indices: Iterable[int]) -> list[str]:
    refs = []
    for i in indices:
        refs += [f"cat:{i}", f"wedge:0:{i}"] + ([f"pct:{i}"] if scene.style.value_labels else [])
    return refs


class PieShare(ChartTemplate):
    id, skill = "pie_share", "retrieval"
    chart_types = PIE

    def instantiate(self, scene, layout, rng):
        i = rng.randrange(len(scene.categories))
        text = f"What percentage of the total does {scene.categories[i]} account for?"
        return self._question(text, {"index": i}, _pie_refs(scene, [i]))

    def solve(self, scene, params, layout):
        i = params["index"]
        value = scene.series[0].values[i]
        how = "read the printed percentage" if scene.style.value_labels else "judge the slice's angle"
        answer = Answer(value=_num(scene, value), type="number", tolerance=_pie_tolerance(scene), unit="%")
        return Solution(answer, [f"find {scene.categories[i]}", f"{how} → {_fmt(scene, value)}%"])


class PieExtremum(ChartTemplate):
    id, skill = "pie_extremum", "extremum"
    chart_types = PIE

    def instantiate(self, scene, layout, rng):
        kind = rng.choice(("largest", "smallest"))
        text = f"Which {_domain(scene).entity[0]} has the {kind} share?"
        return self._question(text, {"kind": kind}, _pie_refs(scene, range(len(scene.categories))))

    def solve(self, scene, params, layout):
        values, cats = scene.series[0].values, scene.categories
        order = sorted(range(len(values)), key=values.__getitem__, reverse=params["kind"] == "largest")
        best, runner_up = order[0], order[1]
        trace = [", ".join(f"{c} {_fmt(scene, v)}%" for c, v in zip(cats, values)), f"{params['kind']} → {cats[best]}"]
        return Solution(Answer(value=cats[best], type="label"), trace, margins=[abs(values[best] - values[runner_up])])


class PieCompare(ChartTemplate):
    id, skill = "pie_compare", "comparison"
    chart_types = PIE

    def instantiate(self, scene, layout, rng):
        values = scene.series[0].values
        a, b = rng.sample(range(len(values)), 2)
        if values[a] == values[b]:
            return None
        if (values[a] > values[b]) != (rng.random() < 0.5):  # keep yes and no balanced
            a, b = b, a
        cats = scene.categories
        return self._question(f"Is {cats[a]}'s share larger than {cats[b]}'s?", {"a": a, "b": b}, _pie_refs(scene, [a, b]))

    def solve(self, scene, params, layout):
        values, cats = scene.series[0].values, scene.categories
        a, b = params["a"], params["b"]
        verdict = "yes" if values[a] > values[b] else "no"
        trace = [f"{cats[a]} {_fmt(scene, values[a])}%, {cats[b]} {_fmt(scene, values[b])}%", f"larger? → {verdict}"]
        return Solution(Answer(value=verdict, type="yes_no"), trace, margins=[abs(values[a] - values[b])])


class PieCombined(ChartTemplate):
    id, skill = "pie_combined", "arithmetic"
    chart_types = PIE

    def instantiate(self, scene, layout, rng):
        a, b = sorted(rng.sample(range(len(scene.categories)), 2))
        cats = scene.categories
        text = f"What is the combined share of {cats[a]} and {cats[b]}? Answer as a percentage."
        return self._question(text, {"a": a, "b": b}, _pie_refs(scene, [a, b]))

    def solve(self, scene, params, layout):
        values, cats = scene.series[0].values, scene.categories
        a, b = params["a"], params["b"]
        total = values[a] + values[b]
        answer = Answer(value=_num(scene, total), type="number", tolerance=2 * _pie_tolerance(scene), unit="%")
        trace = [f"{cats[a]} {_fmt(scene, values[a])}% + {cats[b]} {_fmt(scene, values[b])}%", f"= {_fmt(scene, total)}%"]
        return Solution(answer, trace)


class PieCountAbove(ChartTemplate):
    id, skill = "pie_count_above", "counting"
    chart_types = PIE
    min_categories = 3
    THRESHOLDS = (10, 15, 20, 25, 30, 40)

    def instantiate(self, scene, layout, rng):
        values = scene.series[0].values
        options = [t for t in self.THRESHOLDS if min(values) < t < max(values)]
        if not options:
            return None
        threshold = _balanced_threshold(options, values, rng)
        text = f"How many {_domain(scene).entity[1]} have a share above {threshold}%?"
        return self._question(text, {"threshold": threshold}, _pie_refs(scene, range(len(values))))

    def solve(self, scene, params, layout):
        values, cats = scene.series[0].values, scene.categories
        threshold = params["threshold"]
        above = [cats[i] for i, v in enumerate(values) if v > threshold]
        trace = [f"above {threshold}%: {', '.join(above) or 'none'}", f"count → {len(above)}"]
        return Solution(Answer(value=len(above), type="integer"), trace, margins=[abs(v - threshold) for v in values])


TEMPLATES: list[ChartTemplate] = [
    CountCategories(),
    ReadValue(),
    Extremum(),
    ComparePair(),
    CountAbove(),
    Difference(),
    RankThenRead(),
    WhatIf(),
    TrendDecreases(),
    LargestIncrease(),
    StackTotal(),
    LargestSegment(),
    PieShare(),
    PieExtremum(),
    PieCompare(),
    PieCombined(),
    PieCountAbove(),
]
