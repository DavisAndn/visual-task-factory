"""Dedup: reject images that are near-copies of an image that already has accepted tasks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..core.family import Candidate
from ..core.schema import GateResult


def dhash(image: Image.Image, size: int = 16) -> int:
    """Difference hash: size*size bits that barely change under small visual edits."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    gray = np.asarray(small, dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


class DedupGate:
    name = "dedup"

    def __init__(self, max_hamming: int = 8) -> None:
        self.max_hamming = max_hamming
        self._kept: dict[str, int] = {}

    def check(self, c: Candidate) -> GateResult:
        for key, kept_hash in self._kept.items():
            if key != c.image_key and (kept_hash ^ c.image_hash).bit_count() <= self.max_hamming:
                return GateResult(status="fail", code="near_duplicate", reason=f"the image is a near-copy of {key}")
        return GateResult(status="pass")

    def commit(self, c: Candidate) -> None:
        """Remember an image once one of its tasks is accepted."""
        self._kept[c.image_key] = c.image_hash
