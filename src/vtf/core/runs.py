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


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None
