"""Chart scenes: the data, wording and styling of one image, all sampled from one seeded RNG."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

ChartType = Literal["bar", "hbar", "grouped_bar", "stacked_bar", "line", "pie"]


@dataclass(frozen=True)
class Domain:
    """A topic charts can be about, with the words questions use to talk about it."""

    key: str
    title: str
    entity: tuple[str, str]  # singular, plural
    period: tuple[str, str]
    metric: str
    value_axis: str
    answer_unit: str
    threshold: str  # how a threshold reads in a question, e.g. "{} million"
    low: float
    high: float
    decimals: int
    entities: tuple[str, ...]
    periods: tuple[str, ...]


MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

DOMAINS: tuple[Domain, ...] = (
    Domain(
        key="tourism",
        title="International visitors",
        entity=("country", "countries"),
        period=("year", "years"),
        metric="visitor count",
        value_axis="Visitors (millions)",
        answer_unit="millions",
        threshold="{} million",
        low=1.0,
        high=90.0,
        decimals=1,
        entities=(
            "Japan", "Spain", "Peru", "Italy", "Kenya", "Brazil", "Canada", "Egypt",
            "Norway", "Vietnam", "Mexico", "Greece", "Chile", "Morocco", "Portugal",
        ),
        periods=tuple(str(year) for year in range(2015, 2025)),
    ),
    Domain(
        key="sales",
        title="Revenue",
        entity=("region", "regions"),
        period=("quarter", "quarters"),
        metric="revenue",
        value_axis="Revenue ($M)",
        answer_unit="millions of dollars",
        threshold="${}M",
        low=5.0,
        high=250.0,
        decimals=0,
        entities=("North", "South", "East", "West", "Central", "Coastal", "Metro", "Highlands", "Valley"),
        periods=("Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"),
    ),
    Domain(
        key="energy",
        title="Electricity generation",
        entity=("source", "sources"),
        period=("year", "years"),
        metric="output",
        value_axis="Generation (TWh)",
        answer_unit="TWh",
        threshold="{} TWh",
        low=2.0,
        high=400.0,
        decimals=0,
        entities=("Solar", "Wind", "Hydro", "Nuclear", "Gas", "Coal", "Biomass", "Geothermal"),
        periods=tuple(str(year) for year in range(2014, 2025)),
    ),
    Domain(
        key="apps",
        title="Daily active users",
        entity=("app", "apps"),
        period=("month", "months"),
        metric="user count",
        value_axis="Daily active users (thousands)",
        answer_unit="thousands",
        threshold="{} thousand",
        low=10.0,
        high=900.0,
        decimals=0,
        entities=("Notes", "Maps", "Chat", "Photos", "Music", "Mail", "Weather", "Fitness", "News"),
        periods=MONTHS,
    ),
    Domain(
        key="population",
        title="Population",
        entity=("town", "towns"),
        period=("census year", "census years"),
        metric="population",
        value_axis="Population (thousands)",
        answer_unit="thousands",
        threshold="{} thousand",
        low=5.0,
        high=400.0,
        decimals=0,
        entities=(
            "Ashford", "Brookvale", "Cedarton", "Dunmore", "Elmwood",
            "Fairhaven", "Glenrock", "Harlow", "Ivybridge",
        ),
        periods=("1970", "1980", "1990", "2000", "2010", "2020"),
    ),
)
DOMAIN_BY_KEY = {domain.key: domain for domain in DOMAINS}

PALETTES = {
    "tableau": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"],
    "okabe_ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"],
    "muted": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"],
    "dusk": ["#355C7D", "#C06C84", "#6C5B7B", "#F67280", "#99B898"],
}
BACKGROUNDS = ("#FFFFFF", "#FAFAF7", "#F3F5F8")
LEGEND_LOCS = ("upper right", "upper left", "outside")

DEFAULTS: dict[str, Any] = {
    "chart_types": {"bar": 0.2, "hbar": 0.14, "grouped_bar": 0.16, "stacked_bar": 0.16, "line": 0.18, "pie": 0.16},
    # How many categories (and series) each chart type gets, as [min, max]; capped by the topic's
    # word lists and by the 5 colors in a palette. line_series is a list to pick from.
    "sizes": {
        "bar": [3, 8],
        "hbar": [3, 8],
        "grouped_bar": [3, 6],
        "grouped_series": [2, 3],
        "stacked_bar": [3, 6],
        "stacked_series": [2, 4],
        "line": [5, 9],
        "line_series": [1, 1, 2, 3],
        "pie": [3, 5],
    },
    "style": {
        "value_labels": 0.5,  # probability that exact values are printed on the chart
        # Matplotlib ships these fonts, so images are pixel-identical on any machine.
        "fonts": ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"],
        "font_size": [9.0, 12.0],
        "figsizes": [[6.4, 4.0], [7.2, 4.5], [5.6, 4.2], [6.0, 5.0]],
        "dpi": 110,
        "label_rotations": [0, 0, 30, 45],
        "grid": 0.6,  # probability of gridlines
    },
}
MAX_COLORS = 5  # every palette has five


class Series(BaseModel):
    name: str
    values: list[float]


class ChartStyle(BaseModel):
    font_family: str
    font_size: float
    palette: list[str]
    background: str
    grid: bool
    value_labels: bool
    label_rotation: int
    legend_loc: str
    bar_width: float
    figsize: tuple[float, float]
    dpi: int
    hide_spines: bool
    # Pinned value-axis range. Unset for sampled scenes; contrast pairs set it so that editing one
    # value leaves the axis, gridlines and every other mark exactly where they were.
    value_limits: tuple[float, float] | None = None
    start_angle: int = 90  # pie charts: where the first slice starts, in degrees


class ChartScene(BaseModel):
    family: Literal["charts"] = "charts"
    chart_type: ChartType
    domain: str
    title: str
    category_axis: str
    value_axis: str
    decimals: int
    categories: list[str]
    categories_are_periods: bool  # line charts put periods on the axis and entities in the series
    series: list[Series]
    style: ChartStyle


def _count(rng: random.Random, bounds: list[int], cap: int) -> int:
    low, high = bounds
    return rng.randint(min(low, cap), min(high, cap))


def sample_scene(rng: random.Random, cfg: dict[str, Any]) -> ChartScene:
    chart_type = _weighted(rng, cfg["chart_types"])
    d = rng.choice(DOMAINS)
    sizes = cfg["sizes"]
    if chart_type in ("bar", "hbar"):
        cats = rng.sample(d.entities, _count(rng, sizes[chart_type], len(d.entities)))
        period = rng.choice(d.periods)
        series = [Series(name=period, values=[_value(rng, d) for _ in cats])]
        title = f"{d.title} by {d.entity[0]}, {period}"
    elif chart_type == "grouped_bar":
        cats = rng.sample(d.entities, _count(rng, sizes["grouped_bar"], len(d.entities)))
        k = _count(rng, sizes["grouped_series"], min(len(d.periods), MAX_COLORS))
        start = rng.randrange(len(d.periods) - k + 1)
        periods = d.periods[start : start + k]
        bases = [_value(rng, d) for _ in cats]
        series = [Series(name=p, values=[_drift(rng, base, d) for base in bases]) for p in periods]
        title = f"{d.title} by {d.entity[0]}, {periods[0]}–{periods[-1]}"
    elif chart_type == "stacked_bar":
        # Periods along the axis, each bar split into one segment per entity.
        n = _count(rng, sizes["stacked_bar"], len(d.periods))
        start = rng.randrange(len(d.periods) - n + 1)
        cats = list(d.periods[start : start + n])
        names = rng.sample(d.entities, _count(rng, sizes["stacked_series"], MAX_COLORS))
        series = [Series(name=name, values=[_drift(rng, base, d) for _ in cats]) for name, base in
                  ((name, _value(rng, d)) for name in names)]
        title = f"{d.title} by {d.entity[0]}, {cats[0]}–{cats[-1]}"
    elif chart_type == "pie":
        cats = rng.sample(d.entities, _count(rng, sizes["pie"], min(len(d.entities), MAX_COLORS)))
        period = rng.choice(d.periods)
        decimals = rng.choice((0, 1))
        series = [Series(name=period, values=_shares(rng, len(cats), decimals))]
        title = f"{d.title} by {d.entity[0]}, {period}: share of total"
        return ChartScene(
            chart_type=chart_type,
            domain=d.key,
            title=title,
            category_axis=d.entity[0].capitalize(),
            value_axis="Share of total (%)",
            decimals=decimals,
            categories=list(cats),
            categories_are_periods=False,
            series=series,
            style=_style(rng, cfg["style"], chart_type, 1),
        )
    else:
        n = _count(rng, sizes["line"], len(d.periods))
        start = rng.randrange(len(d.periods) - n + 1)
        cats = list(d.periods[start : start + n])
        names = rng.sample(d.entities, min(rng.choice(sizes["line_series"]), MAX_COLORS))
        series = [Series(name=name, values=_walk(rng, d, n)) for name in names]
        span = f"{cats[0]}–{cats[-1]}"
        title = f"{d.title}: {names[0]}, {span}" if len(names) == 1 else f"{d.title}, {span}"
    on_axis = d.period if chart_type in ("line", "stacked_bar") else d.entity
    return ChartScene(
        chart_type=chart_type,
        domain=d.key,
        title=title,
        category_axis=on_axis[0].capitalize(),
        value_axis=d.value_axis,
        decimals=d.decimals,
        categories=list(cats),
        categories_are_periods=chart_type in ("line", "stacked_bar"),
        series=series,
        style=_style(rng, cfg["style"], chart_type, len(series)),
    )


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = [key for key, weight in weights.items() if weight > 0]
    return rng.choices(keys, weights=[weights[key] for key in keys])[0]


def _value(rng: random.Random, d: Domain) -> float:
    return round(d.low + (d.high - d.low) * rng.random(), d.decimals)


def _drift(rng: random.Random, base: float, d: Domain) -> float:
    return round(min(max(base * (1 + rng.gauss(0, 0.2)), d.low), d.high * 1.15), d.decimals)


def _walk(rng: random.Random, d: Domain, n: int) -> list[float]:
    value = rng.uniform(d.low + 0.25 * (d.high - d.low), d.high * 0.75)
    out = []
    for _ in range(n):
        out.append(round(value, d.decimals))
        value = min(max(value * (1 + rng.gauss(0, 0.12)), d.low), d.high * 1.1)
    return out


def _shares(rng: random.Random, n: int, decimals: int) -> list[float]:
    """n percentages that add up to exactly 100 at the printed precision (largest remainder)."""
    raw = [rng.uniform(0.15, 1.0) for _ in range(n)]
    unit = 10.0 ** -decimals
    exact = [r / sum(raw) * 100 / unit for r in raw]
    counts = [math.floor(x) for x in exact]
    for i in sorted(range(n), key=lambda i: exact[i] - counts[i], reverse=True)[: round(100 / unit) - sum(counts)]:
        counts[i] += 1
    return [round(c * unit, decimals) for c in counts]


def _style(rng: random.Random, cfg: dict[str, Any], chart_type: str, n_series: int) -> ChartStyle:
    # Printed values on multi-series charts often collide, so they are rarer there.
    p_labels = cfg["value_labels"] * (0.5 if n_series > 1 else 1.0)
    low, high = cfg["font_size"]
    return ChartStyle(
        font_family=rng.choice(cfg["fonts"]),
        font_size=round(rng.uniform(low, high) * 2) / 2,
        palette=list(rng.choice(list(PALETTES.values()))),
        background=rng.choice(BACKGROUNDS),
        grid=rng.random() < cfg["grid"],
        value_labels=rng.random() < p_labels,
        label_rotation=0 if chart_type in ("hbar", "pie") else int(rng.choice(cfg["label_rotations"])),
        legend_loc=rng.choice(LEGEND_LOCS),
        bar_width=round(rng.uniform(0.55, 0.85), 2),
        figsize=tuple(rng.choice(cfg["figsizes"])),
        dpi=int(cfg["dpi"]),
        hide_spines=rng.random() < 0.5,
        start_angle=rng.choice((90, 0, 45)) if chart_type == "pie" else 90,
    )
