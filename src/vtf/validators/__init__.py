"""Validation gates. Each takes a candidate task and returns pass, fail (with a reason) or skip."""

from .ambiguity import AmbiguityGate
from .dedup import DedupGate, dhash
from .grounding import GroundingGate

__all__ = ["AmbiguityGate", "DedupGate", "GroundingGate", "dhash"]
