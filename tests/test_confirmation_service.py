"""Tests for the Phase 7A witness confirmation backend."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from confirmation_service import (
    CONFIRMABLE_CATEGORIES,
    apply_confirmed_choices,
    create_confirmation,
    load_confirmation_context,
)


DESCRIPTION = (
    "A young adult male with a round face, short curly black hair, "
    "narrow eyes, thick eyebrows, a broad nose, thin lips, "
    "a thin moustache and a short beard."
)


@pytest.fixture(scope="module")
def confirmation(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("confirmations")
    context = create_confirmation(
        DESCRIPTION,
        "phase7-test-case",
        output_root=output_root,
        top_k=3,
    )
    return context, output_root


def _first_offered_choice(context):
    for category in CONFIRMABLE_CATEGORIES:
        candidates = context["selections"][category]["candidates"]
        if candidates:
            return category, candidates[0]["id"]
    raise AssertionError("Expected at least one offered component candidate")


def test_create_confirmation_rejects_empty_description(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        create_confirmation(
            "   ",
            "empty-description",
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    "unsafe_id",
    ["../escape", "folder/case", r"folder\\case", ""],
)
def test_create_confirmation_rejects_unsafe_id(tmp_path, unsafe_id):
    with pytest.raises((TypeError, ValueError)):
        create_confirmation(
            DESCRIPTION,
            unsafe_id,
            output_root=tmp_path,
        )


def test_create_confirmation_writes_preview_and_context(confirmation):
    context, _ = confirmation

    assert context["confirmation_id"] == "phase7-test-case"
    assert context["description"] == DESCRIPTION
    assert context["confirmed"] is False
    assert Path(context["preview_image"]).is_file()
    assert Path(context["context_file"]).is_file()

    saved = json.loads(
        Path(context["context_file"]).read_text(encoding="utf-8")
    )
    assert saved["confirmation_id"] == context["confirmation_id"]
    assert saved["description"] == DESCRIPTION


def test_confirmation_has_all_review_categories(confirmation):
    context, _ = confirmation

    category_names = [
        category["name"] for category in context["review_categories"]
    ]
    assert category_names == CONFIRMABLE_CATEGORIES
    assert set(context["selections"]) >= set(CONFIRMABLE_CATEGORIES)


def test_candidate_count_respects_top_k(confirmation):
    context, _ = confirmation

    for category in CONFIRMABLE_CATEGORIES:
        candidates = context["selections"][category]["candidates"]
        assert len(candidates) <= 3


def test_load_confirmation_context_round_trip(confirmation):
    context, output_root = confirmation

    loaded = load_confirmation_context(
        context["confirmation_id"],
        output_root=output_root,
    )

    assert loaded["confirmation_id"] == context["confirmation_id"]
    assert loaded["description"] == context["description"]
    assert loaded["analysis"] == context["analysis"]
    assert loaded["selections"] == context["selections"]
    assert Path(loaded["context_file"]).is_file()


def test_load_missing_confirmation_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_confirmation_context(
            "missing-confirmation",
            output_root=tmp_path,
        )


def test_apply_offered_candidate_writes_record_and_image(confirmation):
    context, output_root = confirmation
    category, component_id = _first_offered_choice(context)

    record = apply_confirmed_choices(
        context,
        {category: component_id},
        output_root=output_root,
    )

    assert record["choices"][category] == component_id
    assert Path(record["confirmed_image"]).is_file()
    assert Path(record["record_file"]).is_file()

    saved = json.loads(
        Path(record["record_file"]).read_text(encoding="utf-8")
    )
    assert saved["choices"][category] == component_id


def test_apply_none_removes_optional_component(confirmation):
    context, output_root = confirmation

    record = apply_confirmed_choices(
        context,
        {"beards": "none"},
        output_root=output_root,
    )

    assert record["choices"]["beards"] == "none"
    assert record["plan"]["components"]["beards"] == []
    assert Path(record["confirmed_image"]).is_file()


def test_apply_choices_does_not_mutate_original_plan(confirmation):
    context, output_root = confirmation
    original_plan = deepcopy(context["plan"])
    category, component_id = _first_offered_choice(context)

    apply_confirmed_choices(
        context,
        {category: component_id},
        output_root=output_root,
    )

    assert context["plan"] == original_plan


def test_apply_rejects_non_dictionary_choices(confirmation):
    context, output_root = confirmation

    with pytest.raises(TypeError, match="dictionary"):
        apply_confirmed_choices(
            context,
            [("beards", "none")],
            output_root=output_root,
        )


def test_apply_rejects_unknown_category(confirmation):
    context, output_root = confirmation

    with pytest.raises(ValueError, match="not confirmable"):
        apply_confirmed_choices(
            context,
            {"ears": "ear_01"},
            output_root=output_root,
        )


def test_apply_rejects_component_not_offered(confirmation):
    context, output_root = confirmation

    with pytest.raises(ValueError, match="was not offered"):
        apply_confirmed_choices(
            context,
            {"heads": "head_999"},
            output_root=output_root,
        )
