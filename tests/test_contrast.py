import pytest

from vtf.core.config import load_config
from vtf.core.contrast import answers_differ, make_pairs
from vtf.core.pipeline import run
from vtf.core.runs import read_pairs, read_tasks
from vtf.core.schema import Answer
from vtf.families import get_family

FAMILY = get_family("charts")


@pytest.fixture(scope="module")
def paired_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("paired")
    run(FAMILY, 24, 11, out, load_config(None, FAMILY))
    summary = make_pairs(out)
    return out, summary


def test_numbers_must_differ_by_more_than_both_tolerances():
    assert not answers_differ(Answer(value=50, type="number", tolerance=3), Answer(value=55, type="number", tolerance=3))
    assert answers_differ(Answer(value=50, type="number", tolerance=2), Answer(value=55, type="number", tolerance=2))
    assert answers_differ(Answer(value="yes", type="yes_no"), Answer(value="no", type="yes_no"))
    assert not answers_differ(Answer(value="Peru", type="label"), Answer(value="Peru", type="label"))


def test_pairs_change_one_value_and_flip_the_answer(paired_run):
    out, summary = paired_run
    tasks = {t.id: t for t in read_tasks(out / "tasks.jsonl")}
    pairs = read_pairs(out)
    assert summary["pairs"] == len(pairs) > len(tasks) // 2
    for pair in pairs:
        a, b = tasks[pair.a], pair.b
        assert answers_differ(a.answer, b.answer)
        assert b.question == a.question and all(r.status == "pass" for r in b.validation.values())
        assert (out / b.image).is_file() and b.pixel_hash != a.pixel_hash
        before = [v for s in a.scene["series"] for v in s["values"]]
        after = [v for s in b.scene["series"] for v in s["values"]]
        assert sum(x != y for x, y in zip(before, after)) == 1
        assert b.scene["style"]["value_limits"] is not None  # axis pinned, so nothing else moves


def test_pairs_are_reproducible(paired_run, tmp_path):
    out, _ = paired_run
    first = (out / "pairs.jsonl").read_bytes()
    make_pairs(out)
    assert (out / "pairs.jsonl").read_bytes() == first


def test_count_questions_without_value_edits_are_skipped(paired_run):
    _, summary = paired_run
    row = summary["by_template"].get("count_categories")
    assert row is None or row["pairs"] == 0
