import random

from vtf.core.config import load_config
from vtf.core.pipeline import build_scene, json_hash, pixel_hash
from vtf.families import get_family
from vtf.families.charts.questions import TEMPLATES
from vtf.families.charts.scene import ChartScene, ChartStyle, Series

FAMILY = get_family("charts")
SCENE_CFG = load_config(None, FAMILY)["scene"]
BY_ID = {t.id: t for t in TEMPLATES}


def make_scene(chart_type="bar", values=(30.0, 80.0, 55.0, 10.0), labels=True, rotation=0):
    line = chart_type == "line"
    cats = ["2019", "2020", "2021", "2022", "2023"] if line else ["Japan", "Spain", "Peru", "Italy"]
    style = ChartStyle(
        font_family="DejaVu Sans",
        font_size=10,
        palette=["#4E79A7", "#F28E2B", "#E15759"],
        background="#FFFFFF",
        grid=True,
        value_labels=labels,
        label_rotation=rotation,
        legend_loc="upper right",
        bar_width=0.7,
        figsize=(6.4, 4.0),
        dpi=110,
        hide_spines=False,
    )
    return ChartScene(
        chart_type=chart_type,
        domain="tourism",
        title="International visitors: Japan, 2019–2023" if line else "International visitors by country, 2023",
        category_axis="Year" if line else "Country",
        value_axis="Visitors (millions)",
        decimals=1,
        categories=cats[: len(values)],
        categories_are_periods=line,
        series=[Series(name="Japan" if line else "2023", values=list(values))],
        style=style,
    )


def solve(template_id, scene, **params):
    _, layout = FAMILY.render(scene)
    return BY_ID[template_id].solve(scene, params, layout)


def test_same_seed_gives_identical_scene_and_pixels():
    first = build_scene(FAMILY, 1234, SCENE_CFG)
    second = build_scene(FAMILY, 1234, SCENE_CFG)
    assert json_hash(first[0].model_dump(mode="json")) == json_hash(second[0].model_dump(mode="json"))
    assert pixel_hash(first[1]) == pixel_hash(second[1])


def test_every_chart_type_renders_with_a_full_layout():
    seen = set()
    for seed in range(200):
        scene, image, layout = build_scene(FAMILY, seed, SCENE_CFG)
        if scene.chart_type in seen:
            continue
        seen.add(scene.chart_type)
        assert (layout.width, layout.height) == image.size
        ids = {item.element_id for item in layout.items}
        assert {"title", "axis:value"} <= ids
        assert {f"cat:{i}" for i in range(len(scene.categories))} <= ids
        assert len(layout.find("vtick:*")) >= 2
        if len(seen) == 4:
            break
    assert seen == {"bar", "hbar", "grouped_bar", "line"}


def test_rotated_labels_get_corners_that_fit_their_box():
    _, layout = FAMILY.render(make_scene(rotation=45))
    label = layout.find("cat:1")[0]
    assert label.quad is not None
    xs = [x for x, _ in label.quad]
    ys = [y for _, y in label.quad]
    box = label.bbox
    assert abs(min(xs) - box.x0) < 1.5 and abs(max(xs) - box.x1) < 1.5
    assert abs(min(ys) - box.y0) < 1.5 and abs(max(ys) - box.y1) < 1.5


def test_solvers_on_known_values():
    scene = make_scene(values=(30.0, 80.0, 55.0, 10.0))
    assert solve("extremum", scene, series=0, kind="highest").answer.value == "Spain"
    assert solve("extremum", scene, series=0, kind="lowest").answer.value == "Italy"
    assert solve("compare_pair", scene, series=0, a=0, b=2).answer.value == "no"
    assert solve("count_above", scene, series=0, threshold=40.0).answer.value == 2
    assert solve("difference", scene, series=0, a=1, b=3).answer.value == 70.0
    assert solve("rank_then_read", scene, series=0, rank=2).answer.value == 55.0
    assert solve("what_if", scene, series=0, index=2, factor=2.0).answer.value == "Peru"  # 110 beats 80
    assert solve("what_if", scene, series=0, index=0, factor=2.0).answer.value == "Spain"  # 60 does not
    assert solve("what_if", scene, series=0, index=1, factor=0.5).answer.value == "Peru"  # Spain drops to 40
    assert solve("count_categories", scene).answer.value == 4


def test_trend_counts_decreases():
    scene = make_scene("line", values=(10.0, 12.0, 9.0, 15.0, 11.0))
    assert solve("trend_decreases", scene, series=0).answer.value == 2


def test_values_read_off_the_axis_get_a_tolerance():
    printed = solve("read_value", make_scene(labels=True), series=0, index=1).answer
    axis_only = solve("read_value", make_scene(labels=False), series=0, index=1).answer
    assert printed.tolerance == 0
    assert axis_only.tolerance >= 0.05 * 80.0


def test_questions_depend_on_what_they_mention():
    scene = make_scene(labels=False)
    _, layout = FAMILY.render(scene)
    question = BY_ID["read_value"].instantiate(scene, layout, random.Random(0))
    # The year is only shown in the title, and the value has to be read off the axis.
    assert "title" in question.depends_on and "vtick:*" in question.depends_on
