import json

import pytest

from vtf.core.config import load_config
from vtf.core.pipeline import run
from vtf.core.runs import read_tasks
from vtf.evidence.audit import build_audit, import_audit, wilson
from vtf.families import get_family
from vtf.report import rebuild_report

FAMILY = get_family("charts")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("audited")
    run(FAMILY, 20, 9, out, load_config(None, FAMILY))
    return out


def test_wilson_interval():
    low, high = wilson(100, 100)
    assert high == pytest.approx(1.0) and 0.96 < low < 0.97  # 100/100 still leaves a few percent of doubt
    low, high = wilson(0, 0)
    assert (low, high) == (0.0, 1.0)


def test_audit_page_embeds_the_sample(run_dir):
    page = build_audit(run_dir, n=8).read_text()
    data = json.loads(page.split('<script type="application/json" id="data">')[1].split("</script>")[0])
    assert len(data) == 8 and all(item["image"].startswith("data:image/png;base64,") for item in data)
    assert "__DATA__" not in page and "__KEY__" not in page


def test_imported_results_are_scored(run_dir, tmp_path):
    tasks = read_tasks(run_dir / "tasks.jsonl")[:6]
    items = [
        {"id": t.id, "source": "random", "reviewer_answer": str(t.answer.value), "verdict": "valid"} for t in tasks[:5]
    ]
    items.append({"id": tasks[5].id, "source": "random", "reviewer_answer": "no idea", "verdict": "ambiguous", "note": "can't tell"})
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"reviewer": "tester", "items": items}))
    summary = import_audit(run_dir, results)
    r = summary["random"]
    assert (r["audited"], r["valid"], r["ambiguous"], r["reviewer_agreed"]) == (6, 5, 1, 5)
    assert summary["problems"][0]["id"] == tasks[5].id
    assert "Human audit" in rebuild_report(run_dir).read_text()
