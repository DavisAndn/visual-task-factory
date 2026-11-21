"""Contrast pairs: the same question about a minimally edited image, with a different answer.

A model that answers from priors or from the question text gives the same answer to both images;
one that reads the image gets both right. The family proposes single-value edits, smallest first
(see ChartsFamily.perturb); the first edit that changes the answer by more than both tolerances
and passes the grounding and ambiguity gates becomes the pair. Families without `perturb` are
skipped. Everything is computed from data, so pairs are as reproducible as the tasks themselves.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..families import get_family
from ..validators import AmbiguityGate, GroundingGate
from .family import Candidate, Question
from .pipeline import json_hash, pixel_hash
from .runs import read_tasks
from .schema import Answer, GateResult, Pair, Task


def answers_differ(a: Answer, b: Answer) -> bool:
    """True when no single reading can be right for both answers."""
    if a.type in ("integer", "number") and b.type in ("integer", "number"):
        return abs(float(a.value) - float(b.value)) > a.tolerance + b.tolerance
    return a.value != b.value


def make_pairs(
    run_dir: Path,
    share: float = 1.0,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Build contrast pairs for a share of a run's tasks; write pairs.jsonl and pairs_summary.json."""
    cfg = json.loads((run_dir / "config.json").read_text())
    v = cfg["validation"]
    gates = [
        GroundingGate(v["min_font_px"], v["min_mark_px"], v["overlap_tol_px"], v["clean_text"]),
        AmbiguityGate(v["visual_gap"], v["max_relative_tolerance"]),
    ]
    rng = random.Random(seed)
    tasks = [t for t in read_tasks(run_dir / "tasks.jsonl") if rng.random() < share]
    tried: Counter[str] = Counter()
    made: Counter[str] = Counter()
    sizes: list[float] = []

    with (run_dir / "pairs.jsonl").open("w") as out:
        for task in tasks:
            tried[task.template] += 1
            pair = _pair_for(task, run_dir, gates)
            if pair is not None:
                made[task.template] += 1
                sizes.append(pair.size)
                out.write(pair.model_dump_json() + "\n")
            if progress:
                progress(1)

    summary = {
        "share": share,
        "seed": seed,
        "tasks": len(tasks),
        "pairs": sum(made.values()),
        "median_edit": round(statistics.median(sizes), 3) if sizes else None,
        "by_template": {t: {"tasks": tried[t], "pairs": made[t]} for t in sorted(tried)},
    }
    (run_dir / "pairs_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _pair_for(task: Task, run_dir: Path, gates: list[Any]) -> Pair | None:
    family = get_family(task.family)
    perturb = getattr(family, "perturb", None)
    if perturb is None:
        return None
    template = next(t for t in family.templates if t.id == task.template)
    scene = family.scene_from_dict(task.scene)
    _, layout = family.render(scene)
    question = Question(task.template, task.skill, task.question, task.params, task.depends_on)

    for change, size, edited in perturb(scene, layout, task.params):
        # Screen with the original layout first (cheap); only render edits that flip the answer.
        if not answers_differ(task.answer, template.solve(edited, task.params, layout).answer):
            continue
        image, edited_layout = family.render(edited)
        solution = template.solve(edited, task.params, edited_layout)
        if not answers_differ(task.answer, solution.answer):
            continue
        key = f"images/{task.id}-b.png"
        candidate = Candidate(edited, edited_layout, question, solution, key, 0)
        results: dict[str, GateResult] = {gate.name: gate.check(candidate) for gate in gates}
        if any(result.status == "fail" for result in results.values()):
            continue
        image.save(run_dir / key)
        scene_data = edited.model_dump(mode="json")
        twin = Task(
            id=f"{task.id}-b",
            family=task.family,
            seed=task.seed,
            image=key,
            scene_hash=json_hash(scene_data),
            pixel_hash=pixel_hash(image),
            template=task.template,
            skill=task.skill,
            question=task.question,
            answer=solution.answer,
            params=task.params,
            depends_on=task.depends_on,
            trace=solution.trace,
            validation=results,
            scene=scene_data,
        )
        return Pair(id=f"pair-{task.id}", a=task.id, b=twin, change=change, size=round(size, 4))
    return None
