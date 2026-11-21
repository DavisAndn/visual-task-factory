"""Cross-machine reproducibility: a fixed run must produce exactly the bytes recorded in reference.json.

tasks.jsonl carries every image's pixel hash and pairs.jsonl every twin's, so matching file hashes
mean identical scenes, questions, answers and pixels. CI runs this on Linux against hashes recorded
on macOS. After an intentional change to generation, refresh the reference with:

    VTF_UPDATE_REFERENCE=1 uv run pytest tests/test_reproducible.py
"""

import hashlib
import json
import os
from pathlib import Path

from vtf.core.config import load_config
from vtf.core.contrast import make_pairs
from vtf.core.pipeline import run
from vtf.families import get_family

REFERENCE = Path(__file__).with_name("reference.json")
FAMILY = get_family("charts")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_run_matches_the_recorded_hashes(tmp_path):
    run(FAMILY, 40, 42, tmp_path, load_config(None, FAMILY))
    make_pairs(tmp_path)
    got = {"charts-seed42-n40/tasks.jsonl": _sha(tmp_path / "tasks.jsonl"), "charts-seed42-n40/pairs.jsonl": _sha(tmp_path / "pairs.jsonl")}
    if os.environ.get("VTF_UPDATE_REFERENCE"):
        REFERENCE.write_text(json.dumps(got, indent=2) + "\n")
    expected = json.loads(REFERENCE.read_text())
    assert got == expected, "output changed; if that was intended, refresh tests/reference.json (see this file's docstring)"
