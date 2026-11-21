"""`vtf models`: ask a model every sampled task, with and without its image.

What comes out of it:
- Blind baseline. Open-ended questions a model answers correctly without the image don't test
  vision; they are listed as guessable.
- With the image. Accuracy per template and skill shows what the dataset finds hard. Tasks the
  model gets wrong are listed as unsolved and go to the front of the human audit: they are
  either hard or wrong, and only a person can tell which.
- Contrast pairs (from `vtf pairs`). The edited twin of each sampled task is asked as well; a
  model that really reads the image gets both right.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..clients import cost_usd
from ..core.runs import read_json, read_pairs, read_tasks
from ..core.schema import Task
from ..families import get_family
from ..validators.models import (
    ClientConfigError,
    ModelClient,
    Reply,
    blind_prompt,
    final_answer,
    image_prompt,
    score_response,
)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def reading(reply: str) -> float | None:
    """The number a reply gives as its final answer, if any."""
    match = _NUMBER.search(final_answer(reply).replace("−", "-"))
    return float(match.group(0).replace(",", "")) if match else None

KINDS = ("image", "blind", "pair")


def sample_tasks(tasks: list[Task], n: int, seed: int) -> list[Task]:
    """n tasks spread evenly across templates, in run order (all of them if n is 0 or too big)."""
    if n <= 0 or n >= len(tasks):
        return list(tasks)
    rng = random.Random(seed)
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.template].append(task)
    for group in groups.values():
        rng.shuffle(group)
    picked: set[str] = set()
    while len(picked) < n:
        for template in sorted(groups):
            if groups[template] and len(picked) < n:
                picked.add(groups[template].pop().id)
    return [t for t in tasks if t.id in picked]


def count_jobs(run_dir: Path, n_clients: int, sample: int, seed: int) -> int:
    """How many requests run_model_checks will make (before the cache answers any of them)."""
    chosen = sample_tasks(read_tasks(run_dir / "tasks.jsonl"), sample, seed)
    paired = {pair.a for pair in read_pairs(run_dir)}
    return n_clients * sum(2 + (task.id in paired) for task in chosen)


def _ask(client: ModelClient, task: Task, kind: str, run_dir: Path) -> Reply:
    if kind == "blind":
        return client.ask(blind_prompt(task), None)
    return client.ask(image_prompt(task), run_dir / task.image)


def run_model_checks(
    run_dir: Path,
    clients: list[ModelClient],
    sample: int = 0,
    seed: int = 0,
    workers: int = 8,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Ask every client every sampled task; write model_checks.jsonl and model_summary.json.

    Results for other models from earlier runs are kept, so models can be checked one at a time.
    """
    tasks = read_tasks(run_dir / "tasks.jsonl")
    chosen = sample_tasks(tasks, sample, seed)
    pairs = {pair.a: pair for pair in read_pairs(run_dir)}

    jobs: list[tuple[ModelClient, Task, str]] = []
    for client in clients:
        for task in chosen:
            jobs += [(client, task, "image"), (client, task, "blind")]
            if task.id in pairs:
                jobs.append((client, pairs[task.id].b, "pair"))

    replies: dict[tuple[str, str, str], Reply] = {}
    errors: dict[tuple[str, str, str], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ask, c, t, kind, run_dir): (c.name, t.id, kind) for c, t, kind in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                replies[key] = future.result()
            except ClientConfigError:
                pool.shutdown(cancel_futures=True)
                raise
            except Exception as err:  # one failed request shouldn't sink the run; it's counted
                errors[key] = f"{type(err).__name__}: {err}"
            if progress:
                progress(1)

    prior = (read_json(run_dir / "summary.json") or {}).get("answer_prior", {})
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for client in clients:
        mine = []
        for task in chosen:
            record: dict[str, Any] = {
                "model": client.name,
                "task": task.id,
                "template": task.template,
                "skill": task.skill,
                "facets": get_family(task.family).facets(task.scene),
                "answer": task.answer.model_dump(),
            }
            for kind in KINDS:
                target = pairs[task.id].b if kind == "pair" and task.id in pairs else task
                if kind == "pair" and task.id not in pairs:
                    continue
                key = (client.name, target.id, kind)
                if key in replies:
                    reply = replies[key]
                    record[kind] = {
                        "task": target.id,
                        "reply": reply.text,
                        "correct": score_response(target.answer, reply.text),
                        "served_by": reply.model,
                        "stop_reason": reply.stop_reason,
                        "cost_usd": round(cost_usd(reply), 6),
                        "cached": reply.cached,
                    }
                elif key in errors:
                    record[kind] = {"task": target.id, "error": errors[key]}
            mine.append(record)
        records += mine
        summaries[client.name] = summarize(mine, prior, requested_model=getattr(client, "model", None))

    previous = read_json(run_dir / "model_summary.json") or {}
    models = {k: v for k, v in previous.get("models", {}).items() if k not in summaries}
    models.update(summaries)
    summary = {"sample": len(chosen), "of": len(tasks), "seed": seed, "models": models}
    (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))

    kept = []
    checks_path = run_dir / "model_checks.jsonl"
    if checks_path.is_file():
        kept = [line for line in checks_path.read_text().splitlines() if json.loads(line)["model"] not in summaries]
    checks_path.write_text("".join(line + "\n" for line in kept) + "".join(json.dumps(r) + "\n" for r in records))
    return summary


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def summarize(records: list[dict[str, Any]], prior: dict[str, Any], requested_model: str | None = None) -> dict[str, Any]:
    def correct(record: dict[str, Any], kind: str) -> bool:
        return bool(record.get(kind, {}).get("correct"))

    answered = [r for r in records if "reply" in r.get("image", {})]
    blind = [r for r in records if "reply" in r.get("blind", {})]
    open_blind = [r for r in blind if r["answer"]["type"] != "yes_no"]
    paired = [r for r in records if "reply" in r.get("pair", {}) and "reply" in r.get("image", {})]

    by: dict[str, dict[str, Any]] = {}
    facet_names = ["template", "skill"] + sorted({f for r in records for f in r.get("facets", {})})
    for facet in facet_names:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            value = record[facet] if facet in ("template", "skill") else record.get("facets", {}).get(facet)
            if value is not None:
                groups[str(value)].append(record)
        table = {}
        for name, group in sorted(groups.items()):
            img = [r for r in group if "reply" in r.get("image", {})]
            bl = [r for r in group if "reply" in r.get("blind", {})]
            row: dict[str, Any] = {
                "tasks": len(group),
                "with_image": _rate(sum(correct(r, "image") for r in img), len(img)),
                "blind": _rate(sum(correct(r, "blind") for r in bl), len(bl)),
            }
            if facet == "template" and name in prior:
                row["always_guess"] = prior[name]["share"]
            table[name] = row
        by[facet] = table

    spent, total_cost, served = 0.0, 0.0, defaultdict(int)
    for record in records:
        for kind in KINDS:
            entry = record.get(kind, {})
            if "reply" in entry:
                total_cost += entry["cost_usd"]
                spent += 0.0 if entry["cached"] else entry["cost_usd"]
                served[entry["served_by"]] += 1
    errors = sum("error" in record.get(kind, {}) for record in records for kind in KINDS)
    fallback = sum(n for model, n in served.items() if requested_model and model != requested_model)

    # Reading precision: how far off the model's numbers are when values must be read rather than
    # copied (unprinted: the answer carries a tolerance), whether or not they fall within it.
    misses = []
    for record in answered:
        answer = record["answer"]
        if answer["type"] == "number" and answer["tolerance"] > 0 and float(answer["value"]) != 0:
            got = reading(record["image"]["reply"])
            if got is not None:
                misses.append(abs(got - float(answer["value"])) / abs(float(answer["value"])))
    precision = None
    if misses:
        ranked = sorted(misses)
        precision = {
            "readings": len(ranked),
            "median_error": round(statistics.median(ranked), 4),
            "p90_error": round(ranked[min(len(ranked) - 1, int(0.9 * len(ranked)))], 4),
            "max_error": round(ranked[-1], 4),
        }
    return {
        "tasks": len(records),
        "with_image": {"correct": sum(correct(r, "image") for r in answered), "of": len(answered)},
        "blind": {"correct": sum(correct(r, "blind") for r in blind), "of": len(blind)},
        "blind_open_ended": {"correct": sum(correct(r, "blind") for r in open_blind), "of": len(open_blind)},
        "pairs": {"both_correct": sum(correct(r, "image") and correct(r, "pair") for r in paired), "of": len(paired)},
        "by_template": by.pop("template"),
        "by_skill": by.pop("skill"),
        "by_facet": by,  # image properties: chart type, printed values, rotation, ...
        "guessable": [r["task"] for r in open_blind if correct(r, "blind")],
        "unsolved": [r["task"] for r in answered if not correct(r, "image")],
        "reading_precision": precision,  # relative error on unprinted numeric answers
        "errors": errors,
        "answered_by_fallback": fallback,
        "cost_usd": round(total_cost, 4),
        "spent_this_run_usd": round(spent, 4),
    }
