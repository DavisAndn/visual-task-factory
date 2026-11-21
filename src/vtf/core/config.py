"""Run configuration: defaults in code, optionally overridden by a YAML file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .family import Family

VALIDATION_DEFAULTS: dict[str, Any] = {
    "min_font_px": 9.0,  # smallest text counted as readable
    "min_mark_px": 3.0,  # smallest bar or point counted as visible
    "overlap_tol_px": 1.0,  # overlap allowed between two items before it counts
    "clean_text": True,  # reject images where any two pieces of text collide, used by the question or not
    "visual_gap": 0.05,  # without printed values, deciding gaps must exceed this share of the axis
    "max_relative_tolerance": 0.2,  # numbers read off an axis must be readable to within this share
    "dedup_max_hamming": 8,  # image-hash distance at or below which two images are near-copies
    "determinism_sample": 0.1,  # share of scenes rebuilt from their seed to confirm identical output
}


BALANCE_DEFAULTS: dict[str, Any] = {
    # Skip (don't reject) a task whose answer value would exceed this share of its template's
    # accepted tasks, so always guessing a template's most common answer stays near chance.
    "max_answer_share": 0.35,
    "max_yes_no_share": 0.55,  # yes/no is balanced by construction; this only trims drift
    "warmup": 8,  # accepted tasks per template before the caps apply
}


def load_config(path: Path | None, family: Family) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "family": family.name,
        "questions_per_scene": 3,
        "max_scenes": None,
        "templates": {t.id: t.weight for t in family.templates},
        "scene": copy.deepcopy(family.defaults),
        "validation": dict(VALIDATION_DEFAULTS),
        "balance": dict(BALANCE_DEFAULTS),
    }
    if path is not None:
        overrides = yaml.safe_load(Path(path).read_text()) or {}
        overrides.pop("family", None)
        _merge(cfg, overrides)
    unknown = set(cfg["templates"]) - {t.id for t in family.templates}
    if unknown:
        raise ValueError(f"Unknown templates in config: {', '.join(sorted(unknown))}")
    return cfg


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
