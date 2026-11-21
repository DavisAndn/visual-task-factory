import json
import threading

import pytest

from vtf.clients import CachedClient, ClaudeClient
from vtf.core.config import load_config
from vtf.core.pipeline import run
from vtf.core.runs import read_tasks
from vtf.evidence.models import run_model_checks, sample_tasks
from vtf.families import get_family
from vtf.validators.models import ClientConfigError, Reply

FAMILY = get_family("charts")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    run(FAMILY, 12, 5, out, load_config(None, FAMILY))
    return out


class ScriptedClient:
    """Answers from a lookup table: right with the image, a non-answer without it (or `blind`)."""

    def __init__(self, tasks, name="scripted", correct=True, blind="unsure"):
        self.name = name
        self.answers = {t.question: str(t.answer.value) for t in tasks}
        self.correct = correct
        self.blind = blind
        self.calls = 0
        self._lock = threading.Lock()

    def ask(self, prompt, image_path):
        with self._lock:
            self.calls += 1
        if image_path is None:
            return Reply(text=self.blind, model=self.name)
        question = next(q for q in self.answers if q in prompt)
        return Reply(text=self.answers[question] if self.correct else "wrong", model=self.name, input_tokens=10)


def test_sample_spreads_across_templates(run_dir):
    tasks = read_tasks(run_dir / "tasks.jsonl")
    picked = sample_tasks(tasks, 6, seed=1)
    assert len(picked) == 6
    assert len({t.template for t in picked}) == min(6, len({t.template for t in tasks}))
    assert sample_tasks(tasks, 0, seed=1) == tasks


def test_model_checks_score_and_list_tasks(run_dir):
    tasks = read_tasks(run_dir / "tasks.jsonl")
    right = ScriptedClient(tasks, name="right")
    summary = run_model_checks(run_dir, [right], workers=4)["models"]["right"]
    assert summary["with_image"] == {"correct": len(tasks), "of": len(tasks)}
    assert summary["blind"]["correct"] == 0 and summary["guessable"] == [] and summary["unsolved"] == []
    assert "chart type" in summary["by_facet"]

    wrong = ScriptedClient(tasks, name="wrong", correct=False)
    both = run_model_checks(run_dir, [wrong], workers=4)
    assert set(both["models"]) == {"right", "wrong"}  # earlier models are kept
    assert sorted(both["models"]["wrong"]["unsolved"]) == sorted(t.id for t in tasks)
    lines = (run_dir / "model_checks.jsonl").read_text().splitlines()
    assert len(lines) == 2 * len(tasks)
    assert all(json.loads(line)["image"]["served_by"] in ("right", "wrong") for line in lines)


def test_blind_answers_to_open_questions_are_flagged(run_dir):
    tasks = read_tasks(run_dir / "tasks.jsonl")
    label = next(t for t in tasks if t.answer.type == "label")
    guesser = ScriptedClient(tasks, name="guesser", blind=str(label.answer.value))
    summary = run_model_checks(run_dir, [guesser], workers=4)["models"]["guesser"]
    assert label.id in summary["guessable"]


def test_cache_answers_repeat_requests_for_free(run_dir, tmp_path):
    tasks = read_tasks(run_dir / "tasks.jsonl")
    first = ScriptedClient(tasks, name="cached")
    run_model_checks(run_dir, [CachedClient(first, tmp_path / "cache.jsonl")], workers=4)
    assert first.calls == 2 * len(tasks)
    second = ScriptedClient(tasks, name="cached")
    summary = run_model_checks(run_dir, [CachedClient(second, tmp_path / "cache.jsonl")], workers=4)
    assert second.calls == 0
    assert summary["models"]["cached"]["spent_this_run_usd"] == 0


def test_config_errors_stop_the_run(run_dir):
    class Broken:
        name = "broken"

        def ask(self, prompt, image_path):
            raise ClientConfigError("The API key was rejected")

    with pytest.raises(ClientConfigError):
        run_model_checks(run_dir, [Broken()], workers=2)


def test_claude_request_puts_the_image_first_and_sends_no_sampling_params(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG fake")
    opus = ClaudeClient("claude-opus-5-5", "high").request("Which bar is tallest?", image)
    content = opus["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[-1] == {"type": "text", "text": "Which bar is tallest?"}
    assert opus["output_config"] == {"effort": "high"} and opus["fallbacks"] == "default"
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(opus)
    haiku = ClaudeClient("claude-haiku-5-5", "low").request("Which bar is tallest?", None)
    assert "betas" not in haiku and len(haiku["messages"][0]["content"]) == 1
    with pytest.raises(ValueError):
        ClaudeClient("claude-opus-5-5", "extreme")


def test_report_shows_the_model_evaluation(run_dir):
    from vtf.report import rebuild_report

    tasks = read_tasks(run_dir / "tasks.jsonl")
    run_model_checks(run_dir, [ScriptedClient(tasks, name="report-model", correct=False)], workers=4)
    page = rebuild_report(run_dir).read_text()
    assert "Model evaluation" in page and "report-model answered: wrong" in page
