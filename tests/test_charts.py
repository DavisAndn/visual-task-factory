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
    for seed in range(400):
        scene, image, layout = build_scene(FAMILY, seed, SCENE_CFG)
        if scene.chart_type in seen:
            continue
        seen.add(scene.chart_type)
        assert (layout.width, layout.height) == image.size
        ids = {item.element_id for item in layout.items}
        assert "title" in ids
        assert {f"cat:{i}" for i in range(len(scene.categories))} <= ids
        if scene.chart_type == "pie":  # no axes: slices and their labels instead
            assert len(layout.find("wedge:*")) == len(scene.categories) and not layout.find("vtick:*")
        else:
            assert "axis:value" in ids and len(layout.find("vtick:*")) >= 2
        if len(seen) == 6:
            break
    assert seen == {"bar", "hbar", "grouped_bar", "stacked_bar", "line", "pie"}


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


def test_pie_shares_add_up_to_100_and_stacks_to_their_totals():
    for seed in range(300):
        scene, _, layout = build_scene(FAMILY, seed, SCENE_CFG)
        if scene.chart_type == "pie":
            assert round(sum(scene.series[0].values), scene.decimals) == 100
            answer = solve("pie_share", scene, index=0).answer
            assert answer.unit == "%" and answer.value == scene.series[0].values[0]
        if scene.chart_type == "stacked_bar":
            total = solve("stack_total", scene, index=0).answer.value
            assert abs(total - sum(s.values[0] for s in scene.series)) < 1e-6
            top_segment = solve("read_value", scene, series=len(scene.series) - 1, index=0).answer
            if not scene.style.value_labels:  # read at both ends of the segment
                single = max(0.05 * abs(top_segment.value), layout.meta["tick_step"] / 10)
                assert abs(top_segment.tolerance - 2 * single) < 1e-6


def test_the_hard_preset_prints_nothing_and_crowds_the_charts():
    from pathlib import Path

    hard = load_config(Path(__file__).parents[1] / "configs" / "charts-hard.yaml", FAMILY)["scene"]
    for seed in range(40):
        scene, _, layout = build_scene(FAMILY, seed, hard)
        assert not scene.style.value_labels and not layout.find("val:*")
        assert len(scene.categories) >= 5
        if scene.chart_type in ("grouped_bar", "stacked_bar", "line"):
            assert len(scene.series) >= 3


def test_what_if_names_the_series_again_and_discounts_a_doubled_lead():
    scene = make_scene(values=(30.0, 80.0, 55.0, 10.0), labels=False)
    _, layout = FAMILY.render(scene)
    question = BY_ID["what_if"].instantiate(scene, layout, random.Random(3))
    assert question.text.endswith("in 2023?")  # "highest" alone is ambiguous on multi-year charts
    doubled = solve("what_if", scene, series=0, index=2, factor=2.0)  # Peru 55 -> 110 vs Spain 80
    assert doubled.answer.value == "Peru" and doubled.margins == [15.0]  # (110 - 80) / 2
    halved = solve("what_if", scene, series=0, index=1, factor=0.5)  # Spain 80 -> 40, behind Peru's 55
    assert halved.margins == [15.0]  # halving shrinks reading error, so no discount


def _stacked(labels):
    style = make_scene().style.model_copy(update={"value_labels": labels})
    return ChartScene(
        chart_type="stacked_bar", domain="energy", title="Electricity generation by source, 2019–2021",
        category_axis="Year", value_axis="Generation (TWh)", decimals=0, categories=["2019", "2020", "2021"],
        categories_are_periods=True,
        series=[Series(name="Solar", values=[100.0, 120.0, 90.0]), Series(name="Wind", values=[60.0, 150.0, 70.0])],
        style=style,
    )


def test_unprinted_stacked_segments_need_twice_the_gap():
    unprinted, printed = _stacked(False), _stacked(True)
    # Wind sits on top of Solar, so comparing its segments is a length judgment.
    assert solve("extremum", unprinted, series=1, kind="highest").margins == [40.0]  # (150 - 70) / 2
    assert solve("extremum", printed, series=1, kind="highest").margins == [80.0]
    assert solve("extremum", unprinted, series=0, kind="highest").margins == [20.0]  # bottom series: shared baseline
    assert solve("largest_segment", unprinted, index=1).margins == [15.0]  # (150 - 120) / 2
