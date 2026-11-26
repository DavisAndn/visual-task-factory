"""The numbers the video shows, read from the committed samples' model results, so the pictures
and the narration can't drift from what the runs found."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
RUNS = ["charts-seed42", "instruments-seed42", "connections-seed42", "counting-seed42"]

# Lab colours: the dark-surface steps of the validated categorical palette (dataviz skill).
LABS = {
    "Anthropic": (217, 89, 38),
    "OpenAI": (57, 135, 229),
    "Google": (25, 158, 112),
    "Alibaba": (213, 81, 129),
}


@dataclass(frozen=True)
class Model:
    key: str  # the name `vtf models` records, e.g. "claude-opus-5-5 (high effort)"
    name: str  # for people, e.g. "Claude Opus 5.5"
    short: str  # e.g. "Opus"
    lab: str

    @property
    def color(self) -> tuple[int, int, int]:
        return LABS.get(self.lab, (150, 150, 150))


def describe(key: str) -> Model:
    model_id = key.split(" ")[0].removeprefix("cerebras/").removeprefix("openai/")
    claude = re.match(r"claude-([a-z]+)-(\d+)(?:-(\d+))?$", model_id)
    if claude:
        family, major, minor = claude.groups()
        version = f"{major}.{minor}" if minor else major
        return Model(key, f"Claude {family.capitalize()} {version}", family.capitalize(), "Anthropic")
    if model_id.startswith("gpt-"):
        parts = model_id.removeprefix("gpt-").split("-")
        tier = " ".join(p.capitalize() for p in parts[1:])
        return Model(key, f"GPT-{parts[0]} {tier}".strip(), f"GPT {tier}".strip() if tier else f"GPT-{parts[0]}", "OpenAI")
    if model_id.startswith("gemma"):
        pretty = model_id.replace("gemma-", "Gemma ").replace("-", " ").upper().replace("GEMMA", "Gemma")
        return Model(key, pretty, "Gemma", "Google")
    if model_id.startswith("qwen"):
        pretty = "Qwen " + " ".join(p.upper() if p[-1] == "b" else p for p in model_id.split("-")[1:])
        return Model(key, pretty, "Qwen", "Alibaba")
    return Model(key, model_id, model_id, "Other")


@lru_cache(maxsize=None)
def summary(run: str) -> dict:
    return json.loads((SAMPLES / run / "model_summary.json").read_text())


@lru_cache(maxsize=None)
def checks(run: str) -> dict[tuple[str, str], dict]:
    out = {}
    for line in (SAMPLES / run / "model_checks.jsonl").read_text().splitlines():
        record = json.loads(line)
        out[(record["model"], record["task"])] = record
    return out


def models() -> list[Model]:
    """Every model evaluated on all four runs, by lab and then by price tier as listed in LABS."""
    keys = set(summary(RUNS[0])["models"])
    for run in RUNS[1:]:
        keys &= set(summary(run)["models"])
    order = {"haiku": 0, "sonnet": 1, "opus": 2}
    listed = [describe(k) for k in keys]
    return sorted(listed, key=lambda m: (list(LABS).index(m.lab) if m.lab in LABS else 9, order.get(m.short.lower(), 5), m.name))


def overall(model: Model) -> dict[str, float]:
    """Accuracy over all four runs: with the image, and clean vs photographed on the same tasks."""
    clean = of = p_clean = p_photo = p_of = blind = blind_of = 0
    for run in RUNS:
        m = summary(run)["models"][model.key]
        clean += m["with_image"]["correct"]
        of += m["with_image"]["of"]
        p = m["photos"]
        p_clean, p_photo, p_of = p_clean + p["clean_correct"], p_photo + p["photo_correct"], p_of + p["of"]
        blind += m["blind_open_ended"]["correct"]
        blind_of += m["blind_open_ended"]["of"]
    return {
        "clean": clean / of, "tasks": of, "photo_clean": p_clean / p_of, "photo": p_photo / p_of, "photo_tasks": p_of,
        "blind": blind / blind_of if blind_of else 0.0,
    }


def facet(run: str, name: str, value: str, model: Model) -> dict:
    return summary(run)["models"][model.key]["by_facet"].get(name, {}).get(value, {})


def levels(run: str, prefix: str) -> list[str]:
    found = {v for m in summary(run)["models"].values() for v in m["by_facet"].get("difficulty", {}) if v.startswith(prefix)}
    return sorted(found, key=lambda v: int(v.rsplit("L", 1)[1]))


def answers(run: str, task: str, model: Model) -> dict[str, tuple[str, bool]]:
    """A model's final answers to one task, clean and photographed: (answer text, right?)."""
    from vtf.validators.models import final_answer

    record = checks(run)[(model.key, task)]
    out = {}
    for kind in ("image", "photo"):
        entry = record.get(kind, {})
        out[kind] = (final_answer(entry.get("reply", "")).strip("*").strip(), bool(entry.get("correct")))
    return out


# ---------------------------------------------------------------------------------------------
# Comparing models on the same tasks. Claude answered every task; the other models a labelled
# sample of them, so cross-model numbers are computed on the tasks every model answered.


def common(run: str) -> list[str]:
    """Ids of the tasks every model answered with the image, in run order."""
    keys = [m.key for m in models()]
    answered: dict[str, set[str]] = {}
    for (model, task), record in checks(run).items():
        if "reply" in record.get("image", {}):
            answered.setdefault(task, set()).add(model)
    return [t for t, who in answered.items() if all(k in who for k in keys)]


def tally(run: str, model: Model, kind: str, tasks: list[str], where=None) -> tuple[int, int]:
    """(right, asked) for one kind of check over the given tasks, optionally filtered by facets."""
    right = asked = 0
    for task in tasks:
        record = checks(run).get((model.key, task))
        if not record or (where and not where(record["facets"])):
            continue
        entry = record.get(kind, {})
        if "reply" in entry:
            asked += 1
            right += bool(entry["correct"])
    return right, asked


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if not n:
        return 0.0, 1.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def same_tasks(model: Model) -> dict[str, float | int]:
    """Clean and photographed accuracy over the tasks every model answered, in all four runs;
    photos are compared on the photographed ones."""
    clean = photo_clean = photo = n = n_photo = 0
    for run in RUNS:
        tasks = common(run)
        r, a = tally(run, model, "image", tasks)
        clean, n = clean + r, n + a
        for task in tasks:
            record = checks(run)[(model.key, task)]
            if "reply" in record.get("photo", {}):
                n_photo += 1
                photo_clean += bool(record["image"]["correct"])
                photo += bool(record["photo"]["correct"])
    return {"clean": clean / n, "tasks": n, "photo_clean": photo_clean / n_photo, "photo": photo / n_photo, "photo_tasks": n_photo}


def hook_task(run: str = "instruments-seed42") -> str:
    """The syringe reading for the cold open: one every model read right clean, that the most
    models then misread in its photo. Ties go to the 10 mL syringe, whose scale reads best on screen."""
    best, best_key = None, None
    for task in common(run):
        records = [checks(run).get((m.key, task)) for m in models()]
        if not all(records) or records[0]["template"] != "syringe_read":
            continue
        if not all(r["image"].get("correct") and "reply" in r.get("photo", {}) for r in records):
            continue
        wrong = sum(not r["photo"]["correct"] for r in records)
        key = (wrong, records[0]["facets"].get("difficulty") == "syringe L2")
        if wrong and (best_key is None or key > best_key):
            best, best_key = task, key
    if best is None:
        raise LookupError("no syringe reading that every model got right clean and some got wrong photographed")
    return best


def same_facet(run: str, test) -> tuple[int, list[tuple[Model, float, float]]]:
    """(tasks, [(model, clean, photo)]) over the shared, photographed tasks whose facets pass `test`."""
    tasks = [t for t in common(run) if test(checks(run)[(models()[0].key, t)]["facets"])
             and all("reply" in checks(run)[(m.key, t)].get("photo", {}) for m in models())]
    rows = []
    for m in models():
        c, n = tally(run, m, "image", tasks)
        p, _ = tally(run, m, "photo", tasks)
        rows.append((m, c / n, p / n))
    return len(tasks), rows
