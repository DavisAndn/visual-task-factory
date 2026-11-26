"""Registered image families. To add one, implement the Family protocol and list it here."""

from __future__ import annotations

from ..core.family import Family
from .charts import ChartsFamily
from .connections import ConnectionsFamily
from .counting import CountingFamily
from .instruments import InstrumentsFamily

FAMILIES: dict[str, Family] = {family.name: family for family in (ChartsFamily(), InstrumentsFamily(), ConnectionsFamily(), CountingFamily())}


def get_family(name: str) -> Family:
    if name not in FAMILIES:
        raise ValueError(f"Unknown family {name!r}. Available: {', '.join(sorted(FAMILIES))}.")
    return FAMILIES[name]
