"""Reading the files a run folder holds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Pair, Task


def read_tasks(path: Path) -> list[Task]:
    if not path.is_file():
        return []
    return [Task.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def read_pairs(run_dir: Path) -> list[Pair]:
    path = run_dir / "pairs.jsonl"
    if not path.is_file():
        return []
    return [Pair.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def read_photos(run_dir: Path) -> dict[str, Task]:
    """Photo twins from `vtf photos`, keyed by the id of the clean task each one copies."""
    return {t.id.removesuffix("-photo"): t for t in read_tasks(run_dir / "photos.jsonl")}


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _without_pixel_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_pixel_hashes(v) for k, v in value.items() if k != "pixel_hash"}
    if isinstance(value, list):
        return [_without_pixel_hashes(v) for v in value]
    return value


def task_content(path: Path) -> list[dict[str, Any]]:
    """What a run's tasks say (scenes, questions, answers, gate results), without pixel hashes:
    antialiasing can differ by a few pixels between machines without changing any task."""
    if not path.is_file():
        return []
    return [_without_pixel_hashes(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def compare_runs(a: Path, b: Path) -> list[str]:
    """Where two run folders hold different tasks or pairs (empty if they hold the same ones)."""
    differences = []
    for name in ("tasks.jsonl", "pairs.jsonl", "photos.jsonl"):
        left, right = task_content(a / name), task_content(b / name)
        if left == right:
            continue
        if len(left) != len(right):
            differences.append(f"{name}: {len(left)} entries vs {len(right)}")
            continue
        first = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
        differences.append(f"{name}: first difference at entry {first + 1} ({left[first].get('id')})")
    return differences
