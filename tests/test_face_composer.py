import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from component_selector import load_catalog
from description_parser import analyze_description
from face_composer import (
    build_composition_plan,
    compose_face,
    create_case_id,
    render_composition_plan,
    validate_case_id,
)


def selected_ids(plan, category):
    return [
        component["id"]
        for component in plan["components"][category]
    ]


def test_build_plan_selects_expected_components():
    analysis = analyze_description(
        "A young man with a round face, narrow eyes, a broad nose, "
        "short curly black hair and a thin moustache."
    )
    plan = build_composition_plan(analysis)

    assert selected_ids(plan, "heads") == ["head_05"]
    assert selected_ids(plan, "hair") == ["hair_08"]
    assert selected_ids(plan, "eyes") == ["eyes_08"]
    assert selected_ids(plan, "noses") == ["nose_06"]
    assert selected_ids(plan, "moustaches") == ["moustache_04"]


def test_unmentioned_optional_components_are_skipped():
    analysis = analyze_description("A person with a round face.")
    plan = build_composition_plan(analysis)

    assert plan["components"]["eyes"] == []
    assert plan["components"]["eyebrows"] == []
    assert plan["components"]["beards"] == []
    assert plan["components"]["glasses"] == []


def test_explicit_no_beard_is_skipped():
    analysis = analyze_description(
        "A clean-shaven man with an oval face."
    )
    plan = build_composition_plan(analysis)

    assert plan["components"]["beards"] == []


def test_prominent_ears_select_pair():
    analysis = analyze_description(
        "A person with an oval face and prominent ears."
    )
    plan = build_composition_plan(analysis)

    assert selected_ids(plan, "ears") == ["ear_03", "ear_04"]


def test_default_head_and_neck_are_transparent_fallbacks():
    analysis = analyze_description("A person was seen near the station.")
    plan = build_composition_plan(analysis)

    assert selected_ids(plan, "heads") == ["head_01"]
    assert selected_ids(plan, "necks") == ["neck_02"]
    assert plan["components"]["heads"][0]["fallback"] is True
    assert plan["components"]["necks"][0]["fallback"] is True
    assert len(plan["warnings"]) == 2


def test_render_plan_creates_620_png(tmp_path):
    analysis = analyze_description(
        "A person with a round face, narrow eyes and short curly hair."
    )
    plan = build_composition_plan(analysis)
    output = tmp_path / "reference.png"

    result = render_composition_plan(plan, output)

    assert output.exists()
    assert result["width"] == 620
    assert result["height"] == 620
    assert len(result["rendered_components"]) >= 4

    with Image.open(output) as image:
        assert image.size == (620, 620)
        assert image.mode == "RGBA"


def test_compose_face_creates_case_artifacts(tmp_path):
    analysis = analyze_description(
        "A young woman with an oval face, almond eyes, "
        "a narrow nose and long straight black hair."
    )

    result = compose_face(
        analysis,
        case_id="test_case_001",
        output_root=tmp_path,
    )

    assert Path(result["reference_image"]).exists()
    assert Path(result["analysis_file"]).exists()
    assert Path(result["selection_file"]).exists()
    assert result["width"] == 620
    assert result["height"] == 620

    saved_analysis = json.loads(
        Path(result["analysis_file"]).read_text(encoding="utf-8")
    )
    assert saved_analysis["original_description"].startswith("A young woman")


def test_existing_case_directory_is_not_overwritten(tmp_path):
    analysis = analyze_description("A person with an oval face.")

    compose_face(
        analysis,
        case_id="existing_case",
        output_root=tmp_path,
    )

    with pytest.raises(FileExistsError):
        compose_face(
            analysis,
            case_id="existing_case",
            output_root=tmp_path,
        )


def test_unsafe_case_id_is_rejected():
    with pytest.raises(ValueError):
        validate_case_id("../unsafe")

    with pytest.raises(ValueError):
        validate_case_id("case with spaces")


def test_generated_case_ids_are_unique():
    first = create_case_id()
    second = create_case_id()

    assert first != second
    assert validate_case_id(first) == first
    assert validate_case_id(second) == second


def test_unsafe_component_path_is_rejected(tmp_path):
    analysis = analyze_description("A person with an oval face.")
    plan = build_composition_plan(analysis)
    unsafe_plan = deepcopy(plan)
    unsafe_plan["components"]["heads"][0]["file"] = "../../outside.png"

    with pytest.raises(ValueError):
        render_composition_plan(
            unsafe_plan,
            tmp_path / "unsafe.png",
        )


def test_rendered_components_follow_catalog_layer_order(tmp_path):
    analysis = analyze_description(
        "A person with an oval face, prominent ears, almond eyes, "
        "a narrow nose and long straight hair."
    )
    plan = build_composition_plan(analysis)
    result = render_composition_plan(
        plan,
        tmp_path / "layered.png",
    )

    order = plan["layer_order"]
    rendered_categories = [
        item["category"]
        for item in result["rendered_components"]
    ]
    indexes = [order.index(category) for category in rendered_categories]

    assert indexes == sorted(indexes)


def test_catalog_canvas_is_620_square():
    catalog = load_catalog()
    assert catalog["canvas"]["width"] == 620
    assert catalog["canvas"]["height"] == 620
