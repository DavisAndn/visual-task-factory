from vtf.core.family import Candidate, Question, Solution
from vtf.core.schema import Answer, BBox, LayoutItem, LayoutManifest, Task
from vtf.validators import AmbiguityGate, GroundingGate
from vtf.validators.grounding import overlap_px
from vtf.validators.models import BlindBaselineGate, Reply, score_response

META = {
    "value_axis": [0.0, 100.0],
    "value_ticks": [0.0, 50.0, 100.0],
    "tick_step": 50.0,
    "value_labels": False,
    "value_precision": 0.1,
}


def text(element_id, x0, y0, x1, y1, label="x"):
    return LayoutItem(element_id=element_id, kind="text", bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), text=label, font_px=12)


def candidate(items, depends_on, answer=None, margins=(), meta=None):
    layout = LayoutManifest(width=400, height=300, items=items, meta=meta or META)
    question = Question(template="t", skill="s", text="?", params={}, depends_on=list(depends_on))
    solution = Solution(answer=answer or Answer(value=1, type="integer"), trace=[], margins=list(margins))
    return Candidate(scene=None, layout=layout, question=question, solution=solution, image_key="a.png", image_hash=0)


def strip(dx):
    """A 50x8 px label rotated 45 degrees, shifted right by dx."""
    return [(dx, 0.0), (dx + 35.36, 35.36), (dx + 29.70, 41.02), (dx - 5.66, 5.66)]


def test_overlap_uses_real_shape_of_rotated_labels():
    # 20 px apart their bounding boxes overlap, but the slanted labels themselves don't.
    assert overlap_px(strip(0), strip(20)) == 0
    assert overlap_px(strip(0), strip(8)) > 1


def test_grounding_passes_a_clean_layout():
    items = [text("cat:0", 10, 10, 50, 22), text("cat:1", 60, 10, 100, 22)]
    assert GroundingGate().check(candidate(items, ["cat:0", "cat:1"])).status == "pass"


def test_grounding_rejects_missing_overlapping_and_clipped_labels():
    gate = GroundingGate(clean_text=False)  # check only what the question depends on
    assert gate.check(candidate([text("cat:0", 10, 10, 50, 22)], ["cat:1"])).code == "missing"
    overlapping = [text("cat:0", 10, 10, 50, 22), text("cat:1", 40, 12, 90, 24)]
    assert gate.check(candidate(overlapping, ["cat:0"])).code == "text_overlap"
    assert gate.check(candidate([text("cat:0", 380, 10, 420, 22)], ["cat:0"])).code == "clipped"


def test_grounding_rejects_label_answers_missing_from_the_image():
    items = [text("cat:0", 10, 10, 50, 22, label="Spain")]
    result = GroundingGate().check(candidate(items, ["cat:0"], answer=Answer(value="Peru", type="label")))
    assert result.code == "answer_not_visible"


def test_grounding_rejects_marks_hidden_by_the_legend():
    items = [
        LayoutItem(element_id="bar:0:0", kind="mark", bbox=BBox(x0=100, y0=50, x1=120, y1=200)),
        LayoutItem(element_id="legend", kind="occluder", bbox=BBox(x0=90, y0=40, x1=200, y1=90)),
    ]
    assert GroundingGate().check(candidate(items, ["bar:0:0"])).code == "occluded"


def test_grounding_rejects_cluttered_images_even_when_the_question_avoids_the_clutter():
    items = [text("cat:0", 10, 10, 50, 22), text("val:0:1", 200, 50, 240, 62), text("val:0:2", 230, 52, 270, 64)]
    assert GroundingGate(clean_text=True).check(candidate(items, ["cat:0"])).code == "cluttered_image"
    assert GroundingGate(clean_text=False).check(candidate(items, ["cat:0"])).status == "pass"


def test_ambiguity_rejects_numbers_that_cannot_be_read_precisely():
    gate = AmbiguityGate(max_relative_tolerance=0.2)
    loose = Answer(value=66, type="number", tolerance=35.0)
    assert gate.check(candidate([], [], answer=loose)).code == "imprecise"
    tight = Answer(value=268, type="number", tolerance=13.4)
    assert gate.check(candidate([], [], answer=tight)).status == "pass"


def test_ambiguity_needs_a_visible_gap_unless_values_are_printed():
    gate = AmbiguityGate(visual_gap=0.05)
    assert gate.check(candidate([], [], margins=[3.0])).status == "fail"  # 3 < 5% of 0..100
    assert gate.check(candidate([], [], margins=[6.0])).status == "pass"
    printed = dict(META, value_labels=True)
    assert gate.check(candidate([], [], margins=[0.1], meta=printed)).status == "pass"
    assert gate.check(candidate([], [], margins=[0.0], meta=printed)).status == "fail"


def test_score_response():
    assert score_response(Answer(value=27.5, type="number", tolerance=1.4), "About 28 million")
    assert not score_response(Answer(value=27.5, type="number", tolerance=1.4), "30")
    assert score_response(Answer(value="Spain", type="label"), " spain. ")
    assert score_response(Answer(value="no", type="yes_no"), "No, it was lower.")
    assert not score_response(Answer(value="yes", type="yes_no"), "No")
    assert score_response(Answer(value="yes", type="yes_no"), "**Yes**")  # models sometimes answer in markdown
    assert score_response(Answer(value=133, type="number"), "**133**")
    assert score_response(Answer(value="Harlow", type="label"), "**Harlow**")


class FakeClient:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def ask(self, prompt, image_path):
        return Reply(text=self.reply, model="fake")


def make_task(answer):
    return Task(
        id="t", family="charts", seed=1, image="a.png", scene_hash="x", pixel_hash="y", template="extremum",
        skill="extremum", question="Which country had the most visitors?", answer=answer, params={},
        depends_on=[], trace=[], validation={}, scene={},
    )


def test_blind_baseline_fails_questions_answerable_without_the_image():
    task = make_task(Answer(value="Spain", type="label"))
    assert BlindBaselineGate(FakeClient("Spain")).check(task).status == "fail"
    assert BlindBaselineGate(FakeClient("Peru")).check(task).status == "pass"
    yes_no = make_task(Answer(value="yes", type="yes_no"))
    assert BlindBaselineGate(FakeClient("yes")).check(yes_no).status == "pass"


def test_scoring_reads_the_final_answer_after_shown_working():
    assert score_response(Answer(value=97, type="number"), "335 - 238 = 97\n\n**97**")
    assert score_response(Answer(value=14.5, type="number"), "18.1-3.6=14.5\n\n<tag>\n</tag>\n\n14.5")
    assert score_response(Answer(value=97, type="number"), "335 - 238 = 97")
    assert score_response(Answer(value="yes", type="yes_no"), "Photos 369, Maps 258.\nYes")
    assert not score_response(Answer(value=5.6, type="number"), "-5.6")  # a signed answer to an absolute question
