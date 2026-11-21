import json

from vtf.core.config import load_config
from vtf.core.pipeline import rebuild_task, run
from vtf.core.schema import Task
from vtf.families import get_family

FAMILY = get_family("charts")


def read_tasks(path):
    return [Task.model_validate_json(line) for line in path.read_text().splitlines()]


def test_small_run_produces_valid_packaged_tasks(tmp_path):
    summary = run(FAMILY, 15, 7, tmp_path, load_config(None, FAMILY))
    tasks = read_tasks(tmp_path / "tasks.jsonl")
    assert summary["accepted"] == len(tasks) == 15
    for task in tasks:
        assert (tmp_path / task.image).is_file()
        assert all(result.status != "fail" for result in task.validation.values())
    for name in ("summary.json", "report.html", "config.json", "rejected.jsonl"):
        assert (tmp_path / name).is_file()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert all(rebuild_task(tasks[0], cfg).values())


def test_same_seed_gives_identical_tasks(tmp_path):
    cfg = load_config(None, FAMILY)
    run(FAMILY, 8, 3, tmp_path / "a", cfg)
    run(FAMILY, 8, 3, tmp_path / "b", cfg)
    assert (tmp_path / "a" / "tasks.jsonl").read_bytes() == (tmp_path / "b" / "tasks.jsonl").read_bytes()


def test_the_config_file_lists_exactly_the_defaults():
    from pathlib import Path

    assert load_config(Path(__file__).parents[1] / "configs" / "charts.yaml", FAMILY) == load_config(None, FAMILY)


def test_answer_balancing_caps_a_value_once_warmed_up():
    from collections import Counter

    from vtf.core.config import BALANCE_DEFAULTS
    from vtf.core.pipeline import over_represented
    from vtf.core.schema import Answer

    two = Answer(value=2, type="integer")
    assert not over_represented(two, Counter({"2": 3, "1": 2}), BALANCE_DEFAULTS)  # still warming up
    assert over_represented(two, Counter({"2": 4, "1": 2, "3": 2}), BALANCE_DEFAULTS)  # 5 > 35% of 9, plus 1
    assert not over_represented(Answer(value=4, type="integer"), Counter({"2": 4, "1": 2, "3": 2}), BALANCE_DEFAULTS)
    yes = Answer(value="yes", type="yes_no")
    assert not over_represented(yes, Counter({"yes": 4, "no": 4}), BALANCE_DEFAULTS)  # one task of slack
    assert over_represented(yes, Counter({"yes": 7, "no": 4}), BALANCE_DEFAULTS)
