"""The factory loop: sample a scene, render it, ask questions, run the gates, package the results."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

from .. import __version__
from ..families import get_family
from ..report import write_report
from ..validators import AmbiguityGate, DedupGate, GroundingGate, dhash
from .family import Candidate, Family, Template
from .schema import Answer, GateResult, LayoutManifest, Task

# "render" runs only for families that can check their own pixels against the scene (check_render).
GATE_ORDER = ("determinism", "render", "grounding", "ambiguity", "dedup")


def scene_seed(run_seed: int, family: str, index: int) -> int:
    """Seed for the index-th scene of a run. Stored on every task, so any task can be rebuilt."""
    digest = hashlib.sha256(f"{family}:{run_seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:6], "big")


def json_hash(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()[:16]


def build_scene(family: Family, seed: int, scene_cfg: dict[str, Any]) -> tuple[Any, Image.Image, LayoutManifest]:
    scene = family.sample_scene(random.Random(seed), scene_cfg)
    image, layout = family.render(scene)
    return scene, image, layout


def pick_templates(
    templates: list[Template],
    scene: Any,
    accepted: Counter[str],
    weights: dict[str, float],
    k: int,
    rng: random.Random,
) -> list[Template]:
    """Up to k templates for this scene, most under-represented first (relative to their weights).

    Counting accepted tasks, not attempts, keeps the final mix on target even when some
    templates get rejected more often than others.
    """
    total_weight = sum(weights.values()) or 1.0
    done = sum(accepted.values()) + 1
    pool = [t for t in templates if weights.get(t.id, 0) > 0 and t.applies(scene)]
    rng.shuffle(pool)
    pool.sort(key=lambda t: accepted[t.id] / done - weights[t.id] / total_weight)
    return pool[:k]


def over_represented(answer: Answer, counts: Counter[str], balance: dict[str, Any]) -> bool:
    """Whether accepting this answer would push its value past the template's cap (see config.py)."""
    total = sum(counts.values())
    if not balance or total < balance["warmup"]:
        return False
    cap = balance["max_yes_no_share"] if answer.type == "yes_no" else balance["max_answer_share"]
    # One task of slack, so small counts aren't forced into strict alternation; it fades as runs grow.
    return counts[str(answer.value)] + 1 > cap * (total + 1) + 1


def run(
    family: Family,
    n_tasks: int,
    seed: int,
    out_dir: Path,
    cfg: dict[str, Any],
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Generate until n_tasks pass every gate (or max_scenes is reached), then write the outputs."""
    vcfg = cfg["validation"]
    gates = [
        GroundingGate(vcfg["min_font_px"], vcfg["min_mark_px"], vcfg["overlap_tol_px"], vcfg["clean_text"]),
        AmbiguityGate(vcfg["visual_gap"], vcfg["max_relative_tolerance"], vcfg.get("min_feature_px", 4.0)),
    ]
    dedup = DedupGate(vcfg["dedup_max_hamming"])
    sample = vcfg["determinism_sample"]
    every = max(1, round(1 / sample)) if sample > 0 else 0
    max_scenes = cfg.get("max_scenes") or max(50, n_tasks * 5)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    accepted: list[Task] = []
    rejected: list[Task] = []
    per_template: Counter[str] = Counter()
    answers: dict[str, Counter[str]] = defaultdict(Counter)
    balance_skips: Counter[str] = Counter()
    determinism = {"checked": 0, "passed": 0}
    started = time.perf_counter()
    index = 0

    with (out_dir / "tasks.jsonl").open("w") as ok_file, (out_dir / "rejected.jsonl").open("w") as bad_file:
        while len(accepted) < n_tasks and index < max_scenes:
            sseed = scene_seed(seed, family.name, index)
            scene, image, layout = build_scene(family, sseed, cfg["scene"])
            scene_data = scene.model_dump(mode="json")
            s_hash, p_hash = json_hash(scene_data), pixel_hash(image)
            image_key = f"images/{family.name}-{index:06d}.png"
            image.save(out_dir / image_key)

            det = GateResult(status="skip")
            if every and index % every == 0:
                again, again_image, _ = build_scene(family, sseed, cfg["scene"])
                same = json_hash(again.model_dump(mode="json")) == s_hash and pixel_hash(again_image) == p_hash
                determinism["checked"] += 1
                determinism["passed"] += int(same)
                det = (
                    GateResult(status="pass")
                    if same
                    else GateResult(
                        status="fail",
                        code="nondeterministic",
                        reason="rebuilding from the seed gave a different scene or image",
                    )
                )

            render_check = getattr(family, "check_render", None)
            drawn = render_check(scene, image, layout) if render_check else None
            image_dhash = dhash(image)
            q_rng = random.Random(f"{sseed}:questions")
            chosen = pick_templates(
                family.templates, scene, per_template, cfg["templates"], cfg["questions_per_scene"], q_rng
            )
            for template in chosen:
                question = template.instantiate(scene, layout, q_rng)
                if question is None:
                    continue
                solution = template.solve(scene, question.params, layout)
                cand = Candidate(scene, layout, question, solution, image_key, image_dhash)
                results: dict[str, GateResult] = {"determinism": det}
                if drawn is not None:
                    results["render"] = drawn
                results.update({gate.name: gate.check(cand) for gate in gates})
                results["dedup"] = dedup.check(cand)
                task = Task(
                    id=f"{family.name}-{index:06d}-{template.id}",
                    family=family.name,
                    seed=sseed,
                    image=image_key,
                    scene_hash=s_hash,
                    pixel_hash=p_hash,
                    template=template.id,
                    skill=template.skill,
                    question=question.text,
                    answer=solution.answer,
                    params=question.params,
                    depends_on=question.depends_on,
                    trace=solution.trace,
                    validation=results,
                    scene=scene_data,
                )
                line = task.model_dump_json() + "\n"
                if all(r.status != "fail" for r in results.values()):
                    if over_represented(solution.answer, answers[template.id], cfg.get("balance", {})):
                        balance_skips[template.id] += 1  # valid, just not needed: not a rejection
                        continue
                    answers[template.id][str(solution.answer.value)] += 1
                    accepted.append(task)
                    ok_file.write(line)
                    per_template[template.id] += 1
                    dedup.commit(cand)
                    if progress:
                        progress(len(accepted))
                    if len(accepted) >= n_tasks:
                        break
                else:
                    rejected.append(task)
                    bad_file.write(line)
            index += 1

    elapsed = time.perf_counter() - started
    summary = _summarize(family, seed, n_tasks, index, elapsed, accepted, rejected, determinism)
    summary["balance_skips"] = dict(balance_skips.most_common())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir, summary, accepted, rejected)
    return summary


def rebuild_task(task: Task, cfg: dict[str, Any]) -> dict[str, bool]:
    """Re-derive a task from its seed and report whether the scene, pixels and answer match."""
    family = get_family(task.family)
    scene, image, layout = build_scene(family, task.seed, cfg["scene"])
    template = next(t for t in family.templates if t.id == task.template)
    solution = template.solve(scene, task.params, layout)
    return {
        "scene": json_hash(scene.model_dump(mode="json")) == task.scene_hash,
        "pixels": pixel_hash(image) == task.pixel_hash,
        "answer": solution.answer == task.answer,
    }


def _summarize(
    family: Family,
    seed: int,
    requested: int,
    scenes: int,
    elapsed: float,
    accepted: list[Task],
    rejected: list[Task],
    determinism: dict[str, int],
) -> dict[str, Any]:
    candidates = len(accepted) + len(rejected)

    first_failure: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for task in rejected:
        for gate in GATE_ORDER:
            result = task.validation.get(gate)
            if result is not None and result.status == "fail":
                first_failure[gate] += 1
                reasons[f"{gate}: {result.code}"] += 1
                break
    funnel, remaining = [], candidates
    ran = [g for g in GATE_ORDER if any(g in t.validation for t in accepted + rejected)]
    for gate in ran:
        remaining -= first_failure[gate]
        funnel.append({"gate": gate, "rejected": first_failure[gate], "remaining": remaining})

    # Shortcuts a model could use without looking: always giving a template's most common
    # answer, or answering with a name that appears in the question itself.
    by_template: dict[str, list[Task]] = defaultdict(list)
    for task in accepted:
        by_template[task.template].append(task)
    prior = {}
    for template, tasks in sorted(by_template.items()):
        top, count = Counter(str(t.answer.value) for t in tasks).most_common(1)[0]
        entry: dict[str, Any] = {"tasks": len(tasks), "most_common_answer": top, "share": round(count / len(tasks), 3)}
        labels = [t for t in tasks if t.answer.type == "label"]
        if labels:
            named = sum(bool(re.search(rf"\b{re.escape(str(t.answer.value))}\b", t.question)) for t in labels)
            entry["named_in_question"] = round(named / len(labels), 3)
        prior[template] = entry

    coverage: dict[str, Counter[str]] = {
        "skill": Counter(t.skill for t in accepted),
        "template": Counter(t.template for t in accepted),
    }
    for task in accepted:
        for facet, value in family.facets(task.scene).items():
            coverage.setdefault(facet, Counter())[value] += 1

    return {
        "family": family.name,
        "seed": seed,
        "requested": requested,
        "accepted": len(accepted),
        "candidates": candidates,
        "acceptance_rate": round(len(accepted) / candidates, 3) if candidates else 0.0,
        "scenes_rendered": scenes,
        "elapsed_seconds": round(elapsed, 2),
        "funnel": funnel,
        "rejection_reasons": dict(reasons.most_common()),
        "coverage": {facet: dict(counts.most_common()) for facet, counts in coverage.items()},
        "answer_prior": prior,
        "determinism": determinism,
        "versions": {
            "vtf": __version__,
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
    }
