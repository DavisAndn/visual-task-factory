"""Reproducibility: a fixed run must produce the tasks recorded in reference.json, on any machine.

The fingerprint covers what each task says (scene, question, answer, gate results) and leaves out
pixel hashes, since antialiasing can differ by a few pixels between machines without changing any
task. After an intentional change to generation, refresh the reference with:

    VTF_UPDATE_REFERENCE=1 uv run pytest tests/test_reproducible.py
"""

import hashlib
import json
import os
from pathlib import Path

from vtf.core.config import load_config
from vtf.core.contrast import make_pairs
from vtf.core.pipeline import run
from vtf.core.runs import task_content
from vtf.families import get_family

REFERENCE = Path(__file__).with_name("reference.json")
FAMILY = get_family("charts")


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(json.dumps(task_content(path), sort_keys=True).encode()).hexdigest()


def test_fixed_run_matches_the_recorded_tasks(tmp_path):
    run(FAMILY, 40, 42, tmp_path, load_config(None, FAMILY))
    make_pairs(tmp_path)
    got = {
        "charts-seed42-n40/tasks": _fingerprint(tmp_path / "tasks.jsonl"),
        "charts-seed42-n40/pairs": _fingerprint(tmp_path / "pairs.jsonl"),
    }
    if os.environ.get("VTF_UPDATE_REFERENCE"):
        REFERENCE.write_text(json.dumps(got, indent=2) + "\n")
    expected = json.loads(REFERENCE.read_text())
    assert got == expected, "the tasks changed; if that was intended, refresh tests/reference.json (see this file's docstring)"


def test_compare_ignores_pixel_hashes_only(tmp_path):
    from vtf.core.runs import compare_runs

    a, b = tmp_path / "a", tmp_path / "b"
    for folder in (a, b):
        folder.mkdir()
    task = {"id": "t1", "question": "Q?", "pixel_hash": "aaaa", "answer": {"value": 3}}
    (a / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (b / "tasks.jsonl").write_text(json.dumps({**task, "pixel_hash": "bbbb"}) + "\n")
    assert compare_runs(a, b) == []
    (b / "tasks.jsonl").write_text(json.dumps({**task, "answer": {"value": 4}}) + "\n")
    assert compare_runs(a, b) == ["tasks.jsonl: first difference at entry 1 (t1)"]
