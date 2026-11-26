"""Instruments, connections, photo twins, and the new answer types.

The connection answers are re-derived here by independent solvers: gear directions from which gears
the drawing shows touching (not from the generator's parent links), valve reachability with
union-find instead of the family's search."""

import json
import math
import random
from pathlib import Path

import pytest

from vtf.core.capture import make_photos, photograph
from vtf.core.config import load_config
from vtf.core.contrast import answers_differ
from vtf.core.pipeline import rebuild_task, run
from vtf.core.runs import read_tasks
from vtf.core.schema import Answer
from vtf.families import get_family
from vtf.families.drawing import probe_check
from vtf.validators.models import score_response

INSTRUMENTS = get_family("instruments")
CONNECTIONS = get_family("connections")
COUNTING = get_family("counting")
CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("family", [INSTRUMENTS, CONNECTIONS, COUNTING], ids=["instruments", "connections", "counting"])
def test_small_run_passes_every_gate_and_rebuilds(family, tmp_path):
    summary = run(family, 30, 11, tmp_path, load_config(None, family))
    tasks = read_tasks(tmp_path / "tasks.jsonl")
    assert summary["accepted"] == len(tasks) == 30
    assert all(r.status != "fail" for t in tasks for r in t.validation.values())
    assert any("render" in t.validation and t.validation["render"].status == "pass" for t in tasks)
    cfg = json.loads((tmp_path / "config.json").read_text())
    for task in tasks[:6]:
        assert all(rebuild_task(task, cfg).values())


@pytest.mark.parametrize("name", ["instruments", "connections", "counting"])
def test_the_config_file_lists_exactly_the_defaults(name):
    family = get_family(name)
    assert load_config(CONFIGS / f"{name}.yaml", family) == load_config(None, family)


def test_readings_sit_on_ticks_between_labels():
    cfg = load_config(None, INSTRUMENTS)["scene"]
    for seed in range(300):
        scene = INSTRUMENTS.sample_scene(random.Random(seed), cfg)
        if scene.kind == "clock":
            assert 0 <= scene.minute < 60 and 1 <= scene.hour <= 12
            continue
        scale = scene.inner or scene.scale
        reading = scene.value / scene.factor if scene.inner else scene.value
        ticks = reading / scale.minor
        assert abs(ticks - round(ticks)) < 1e-6, (seed, reading, scale)
        assert round(ticks) % scale.per_label != 0, "readings avoid labelled ticks, so ticks must be counted"


def _touching(gears, i, j):
    a, b = gears.gears[i], gears.gears[j]
    return abs(math.hypot(a.x - b.x, a.y - b.y) - gears.radius(i) - gears.radius(j)) < 0.05


def test_gear_directions_match_the_touching_gears_in_the_drawing():
    cfg = load_config(None, CONNECTIONS)["scene"]
    template = next(t for t in CONNECTIONS.templates if t.id == "gears_ccw")
    checked = 0
    for seed in range(400):
        scene = CONNECTIONS.sample_scene(random.Random(seed), cfg)
        if scene.kind != "gears":
            continue
        g = scene.gears
        side, todo = {0: 0}, [0]  # 2-colour the contact graph from the driver
        while todo:
            i = todo.pop()
            for j in range(len(g.gears)):
                if j != i and _touching(g, i, j):
                    assert side.get(j, 1 - side[i]) == 1 - side[i], "a loop of touching gears would jam"
                    if j not in side:
                        side[j] = 1 - side[i]
                        todo.append(j)
        ccw = {g.gears[i].label for i, s in side.items() if s == 1}
        answer = template.solve(scene, {}, None).answer
        assert set(str(answer.value).split(", ")) - {"none"} == ccw
        checked += 1
    assert checked > 50


def test_valve_reachability_matches_union_find():
    cfg = load_config(None, CONNECTIONS)["scene"]
    checked = 0
    for seed in range(400):
        scene = CONNECTIONS.sample_scene(random.Random(seed), cfg)
        if scene.kind != "valves":
            continue
        v = scene.valves
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p, (a, b) in enumerate(v.pipes):
            if p not in v.closed:
                parent[find(a)] = find(b)
        expect = [k for k in range(len(v.tanks)) if find(f"T{k}") == find("P")]
        assert v.reachable() == expect
        checked += 1
    assert checked > 50


def test_the_pixel_check_catches_a_gear_drawn_out_of_mesh():
    cfg = load_config(None, CONNECTIONS)["scene"]
    scene = next(s for s in (CONNECTIONS.sample_scene(random.Random(seed), cfg) for seed in range(100)) if s.kind == "gears")
    image, layout = CONNECTIONS.render(scene)
    assert probe_check(image, layout).status == "pass"
    moved = scene.model_copy(deep=True)
    child = next(i for i, gear in enumerate(moved.gears.gears) if gear.parent == 0)
    a, c = moved.gears.gears[0], moved.gears.gears[child]
    d = math.hypot(c.x - a.x, c.y - a.y)
    c.x, c.y = a.x + (c.x - a.x) * (d + 4) / d, a.y + (c.y - a.y) * (d + 4) / d  # 4 units further out
    moved_image, _ = CONNECTIONS.render(moved)
    result = probe_check(moved_image, layout)  # the data still says the gears mesh
    assert result.status == "fail" and result.code == "drawing_mismatch"


def test_the_pixel_check_catches_a_needle_in_the_wrong_place():
    cfg = load_config(None, INSTRUMENTS)["scene"]
    scene = next(s for s in (INSTRUMENTS.sample_scene(random.Random(seed), cfg) for seed in range(100)) if s.kind == "gauge")
    _, layout = INSTRUMENTS.render(scene)
    elsewhere = scene.model_copy(update={"value": (scene.value + 0.4 * scene.scale.top) % scene.scale.top})
    image, _ = INSTRUMENTS.render(elsewhere)
    assert probe_check(image, layout).status == "fail"


def test_contrast_edits_flip_the_answer(tmp_path):
    from vtf.core.contrast import make_pairs

    run(INSTRUMENTS, 20, 5, tmp_path, load_config(None, INSTRUMENTS))
    summary = make_pairs(tmp_path)
    assert summary["pairs"] >= 15
    tasks = {t.id: t for t in read_tasks(tmp_path / "tasks.jsonl")}
    from vtf.core.runs import read_pairs

    for pair in read_pairs(tmp_path):
        assert answers_differ(tasks[pair.a].answer, pair.b.answer)


def test_photo_twins_keep_the_answer_and_are_reproducible(tmp_path):
    run(INSTRUMENTS, 16, 9, tmp_path / "run", load_config(None, INSTRUMENTS))
    first = make_photos(tmp_path / "run")
    photos = read_tasks(tmp_path / "run" / "photos.jsonl")
    clean = {t.id: t for t in read_tasks(tmp_path / "run" / "tasks.jsonl")}
    assert first["photos"] == len(photos) >= 12
    for twin in photos:
        original = clean[twin.id.removesuffix("-photo")]
        assert twin.answer == original.answer and twin.question == original.question
        assert twin.capture and (tmp_path / "run" / twin.image).is_file()
    again = (tmp_path / "run" / "photos.jsonl").read_text()
    make_photos(tmp_path / "run")
    assert (tmp_path / "run" / "photos.jsonl").read_text() == again


def test_photograph_maps_labels_onto_the_photo():
    cfg = load_config(None, INSTRUMENTS)["scene"]
    scene = INSTRUMENTS.sample_scene(random.Random(2), cfg)
    image, layout = INSTRUMENTS.render(scene)
    capture = {"corners": [[0.04, -0.03], [-0.05, 0.02], [0.03, 0.05], [-0.02, -0.04]], "shade": [0.5, -0.5, 0.2],
               "glare": [0.5, 0.5, 0.15, 40], "blur": 0.8, "noise": 5.0, "noise_seed": 1, "jpeg_quality": 55}
    photo, mapped = photograph(image, layout, capture)
    assert photo.size == (mapped.width, mapped.height)
    for before, after in zip(layout.items, mapped.items):
        assert after.element_id == before.element_id
        assert after.bbox.inside(mapped.width, mapped.height)
        if before.font_px:
            assert 0.85 < after.font_px / before.font_px < 1.15


def test_set_and_time_answers_are_scored():
    tanks = Answer(value="T2, T10", type="set", options=["T1", "T2", "T3", "T10"])
    assert score_response(tanks, "Tracing the pipes...\nT10, T2")
    assert score_response(tanks, "**T2 and T10**")
    assert not score_response(tanks, "T1, T2, T10")
    assert not score_response(tanks, "T2")
    nothing = Answer(value="none", type="set", options=["B", "C"])
    assert score_response(nothing, "None") and not score_response(nothing, "B")
    clock = Answer(value="12:59", type="time", tolerance=1)
    assert score_response(clock, "1:00") and score_response(clock, "It shows 12:59.")
    assert not score_response(clock, "12:57") and not score_response(clock, "no idea")
    assert answers_differ(Answer(value="3:10", type="time", tolerance=1), Answer(value="3:13", type="time", tolerance=1))
    assert not answers_differ(Answer(value="3:10", type="time", tolerance=1), Answer(value="3:11", type="time", tolerance=1))


def test_counted_points_stay_clear_of_each_other_and_the_line():
    cfg = load_config(None, COUNTING)["scene"]
    for seed in range(12):
        scene = COUNTING.sample_scene(random.Random(seed), cfg)
        lo, hi = {1: (25, 35), 2: (60, 80), 3: (140, 160), 4: (280, 320), 5: (480, 550), 6: (750, 850)}[scene.level]
        assert lo <= len(scene.points) <= hi
        coord = 0 if scene.vertical else 1
        assert all(abs(p[coord] - scene.threshold) > 0.3 for p in scene.points)
        side = "right" if scene.vertical else "above"
        assert scene.count(side) + scene.count("left" if scene.vertical else "below") == len(scene.points)


def test_a_remark_in_brackets_after_the_answer_is_ignored():
    port = Answer(value="A7", type="label")
    assert score_response(port, "**A7** (best estimate; the tightly crossing cables make this trace uncertain).")
    assert not score_response(port, "A8 (or possibly A7)")
    tanks = Answer(value="T2, T5", type="set", options=["T1", "T2", "T3", "T5"])
    assert score_response(tanks, "T2, T5 (T3 is cut off by a closed valve)")
    assert score_response(Answer(value=0.41, type="number", tolerance=0.0045), "(0.41)")
