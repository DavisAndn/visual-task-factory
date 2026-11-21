import pytest
from PIL import Image

from vtf.core.config import load_config
from vtf.core.pipeline import build_scene
from vtf.evidence.ocr import matches, prepare, read_label, tesseract_path
from vtf.families import get_family

FAMILY = get_family("charts")
SCENE_CFG = load_config(None, FAMILY)["scene"]


def test_matching_forgives_look_alike_characters_but_not_different_text():
    assert matches("Q1 2024", "Ql 2024")
    assert matches("Harlow", "HarLow")
    assert matches("52.2", "52.2")
    assert not matches("52", "92")
    assert not matches("Spain", "")


def test_rotated_labels_are_levelled_before_reading():
    for seed in range(200):
        scene, image, layout = build_scene(FAMILY, seed, SCENE_CFG)
        rotated = [i for i in layout.items if i.kind == "text" and i.quad]
        if rotated:
            prepared = prepare(image, rotated[0])
            assert prepared is not None and prepared.width > prepared.height  # lying flat
            return
    pytest.fail("no rotated labels in 200 scenes")


@pytest.mark.skipif(tesseract_path() is None, reason="Tesseract is not installed")
def test_tesseract_reads_back_category_labels():
    read = total = 0
    for seed in range(12):
        _, image, layout = build_scene(FAMILY, seed, SCENE_CFG)
        for item in layout.find("cat:*"):
            total += 1
            read += read_label(image, item)[1] > 0
    assert total and read / total > 0.95
