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


def test_model_ids_pick_the_provider_and_the_request_shape(monkeypatch, tmp_path):
    from vtf.clients import ChatCompletionsClient, key_present, make_client, provider_of

    assert [provider_of(m) for m in ("claude-opus-5-5", "gpt-6.1", "openai/o9", "cerebras/gemma-4-31b")] == [
        "anthropic", "openai", "openai", "cerebras"]
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    assert key_present("cerebras/gemma-4-31b") == "CEREBRAS_API_KEY"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    assert key_present("gpt-6.1") is None and isinstance(make_client("claude-haiku-5-5"), ClaudeClient)
    image = tmp_path / "a.jpg"
    image.write_bytes(b"\xff\xd8 fake")
    gpt = make_client("gpt-6.1", "high")
    sent = gpt.request("Which bar is tallest?", image)
    system, user = sent["messages"]
    assert system["role"] == "system" and sent["model"] == "gpt-6.1" and sent["reasoning_effort"] == "high"
    picture = user["content"][0]["image_url"]
    assert picture["url"].startswith("data:image/jpeg;base64,") and picture["detail"] == "high"
    assert user["content"][-1] == {"type": "text", "text": "Which bar is tallest?"}
    gemma = ChatCompletionsClient("cerebras/gemma-4-31b")
    sent = gemma.request("Which bar is tallest?", None)
    assert sent["model"] == "gemma-4-31b" and "reasoning_effort" not in sent and len(sent["messages"][1]["content"]) == 1
    assert gemma.name == "cerebras/gemma-4-31b" and gpt.name == "gpt-6.1 (high effort)"
    assert gemma.signature != ChatCompletionsClient("cerebras/qwen-3.8-27b").signature


def test_cost_uses_the_longest_matching_price(monkeypatch):
    from vtf import clients

    monkeypatch.setitem(clients.PRICES, "gpt-test", (1.0, 10.0))
    dated = Reply(text="4", model="gpt-test-2026-10-01", input_tokens=1_000_000, output_tokens=100_000)
    assert clients.cost_usd(dated) == pytest.approx(2.0)
    opus = Reply(text="4", model="claude-opus-5-5", input_tokens=1_000_000, output_tokens=0)
    assert clients.cost_usd(opus) == pytest.approx(4.0)  # not claude-opus-5's price
    assert clients.cost_usd(Reply(text="4", model="unknown-model", input_tokens=5)) == 0.0


def test_report_shows_the_model_evaluation(run_dir):
    from vtf.report import rebuild_report

    tasks = read_tasks(run_dir / "tasks.jsonl")
    run_model_checks(run_dir, [ScriptedClient(tasks, name="report-model", correct=False)], workers=4)
    page = rebuild_report(run_dir).read_text()
    assert "Model evaluation" in page and "report-model answered: wrong" in page


def test_fallback_count_sees_through_the_cache_and_dated_names(run_dir, tmp_path):
    tasks = read_tasks(run_dir / "tasks.jsonl")

    class Dated(ScriptedClient):
        api_model = "gpt-test"

        def ask(self, prompt, image_path):
            reply = super().ask(prompt, image_path)
            reply.model = "gpt-test-2026-10-01" if image_path is not None else "other-model"
            return reply

    client = CachedClient(Dated(tasks, name="dated"), tmp_path / "cache.jsonl")
    summary = run_model_checks(run_dir, [client], workers=2)["models"]["dated"]
    blind_replies = summary["blind"]["of"]
    assert summary["answered_by_fallback"] == blind_replies  # the dated snapshot isn't a fallback; "other-model" is


def test_each_model_keeps_its_own_sample(run_dir):
    tasks = read_tasks(run_dir / "tasks.jsonl")
    run_model_checks(run_dir, [ScriptedClient(tasks, name="everything")], workers=2)
    summary = run_model_checks(run_dir, [ScriptedClient(tasks, name="few")], sample=3, workers=2)
    assert summary["models"]["few"]["sample"] == 3
    assert summary["models"]["everything"]["sample"] == len(tasks) == summary["sample"]
