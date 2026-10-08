"""Backend service for witness review and component confirmation."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from component_selector import load_catalog, select_all_components
from description_parser import analyze_description
from face_composer import (
    DEFAULT_CATALOG_PATH,
    build_composition_plan,
    render_composition_plan,
    validate_case_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIRMATION_ROOT = (
    PROJECT_ROOT / "static" / "outputs" / "confirmations"
)

CONFIRMABLE_CATEGORIES = [
    "heads",
    "hair",
    "eyes",
    "eyebrows",
    "noses",
    "lips",
    "moustaches",
    "beards",
]

CATEGORY_LABELS = {
    "heads": "Face shape",
    "hair": "Hair",
    "eyes": "Eyes",
    "eyebrows": "Eyebrows",
    "noses": "Nose",
    "lips": "Lips",
    "moustaches": "Moustache",
    "beards": "Beard",
}


def _confirmation_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id


def _feature_summary(analysis):
    summary = []

    for feature_name, state in analysis["features"].items():
        if state["value"] is None:
            continue

        summary.append(
            {
                "name": feature_name,
                "label": feature_name.replace("_", " ").title(),
                "value": state["value"],
                "display_value": state["value"].replace("_", " ").title(),
                "confidence": state["confidence"],
                "source_text": state["source_text"],
                "negated": state["negated"],
            }
        )

    return summary


def _review_categories(selections):
    categories = []

    for category in CONFIRMABLE_CATEGORIES:
        result = selections[category]

        categories.append(
            {
                "name": category,
                "label": CATEGORY_LABELS[category],
                "requested_attributes": deepcopy(
                    result["requested_attributes"]
                ),
                "explicitly_absent": result["explicitly_absent"],
                "reason": result["reason"],
                "candidates": deepcopy(result["candidates"]),
            }
        )

    return categories


def create_confirmation(
    description,
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
    top_k=3,
):
    """Analyze a description and create an initial review preview."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("A non-empty witness description is required")

    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    confirmation_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(catalog_path)
    analysis = analyze_description(description)
    selections = select_all_components(
        analysis,
        top_k=top_k,
        catalog=catalog,
    )
    plan = build_composition_plan(analysis, catalog=catalog)

    preview_path = confirmation_dir / "preview_reference.png"
    render_result = render_composition_plan(
        plan,
        preview_path,
        component_root=Path(catalog_path).parent,
    )

    context = {
        "confirmation_id": confirmation_id,
        "description": description,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "analysis": analysis,
        "feature_summary": _feature_summary(analysis),
        "selections": selections,
        "review_categories": _review_categories(selections),
        "plan": plan,
        "preview_image": str(preview_path),
        "preview_components": render_result["rendered_components"],
        "warnings": list(plan["warnings"]),
        "confirmed": False,
        "confirmed_choices": {},
    }

    context_path = confirmation_dir / "confirmation_context.json"
    context_path.write_text(
        json.dumps(context, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    context["context_file"] = str(context_path)

    return context


def _allowed_candidate_map(context, category):
    result = context["selections"].get(category)
    if result is None:
        return {}

    return {
        candidate["id"]: candidate
        for candidate in result["candidates"]
    }


def apply_confirmed_choices(
    context,
    choices,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Apply witness choices and render the confirmed reference image.

    choices format:
        {"heads": "head_05", "eyes": "eyes_08", "beards": "none"}
    """
    if not isinstance(choices, dict):
        raise TypeError("choices must be a dictionary")

    confirmation_id = context["confirmation_id"]
    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    confirmation_dir.mkdir(parents=True, exist_ok=True)

    plan = deepcopy(context["plan"])
    applied_choices = {}

    for category, selected_id in choices.items():
        if category not in CONFIRMABLE_CATEGORIES:
            raise ValueError(
                f"Category is not confirmable: {category}"
            )

        if selected_id in (None, "", "none"):
            plan["components"][category] = []
            applied_choices[category] = "none"
            continue

        allowed = _allowed_candidate_map(context, category)

        if selected_id not in allowed:
            raise ValueError(
                f"Component '{selected_id}' was not offered for "
                f"category '{category}'"
            )

        chosen = deepcopy(allowed[selected_id])
        chosen["fallback"] = False
        plan["components"][category] = [chosen]
        applied_choices[category] = selected_id

    confirmed_path = confirmation_dir / "confirmed_reference.png"
    render_result = render_composition_plan(
        plan,
        confirmed_path,
        component_root=Path(catalog_path).parent,
    )

    confirmation_record = {
        "confirmation_id": confirmation_id,
        "description": context["description"],
        "confirmed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "analysis": context["analysis"],
        "choices": applied_choices,
        "plan": plan,
        "confirmed_image": str(confirmed_path),
        "rendered_components": render_result["rendered_components"],
        "warnings": list(plan["warnings"]),
    }

    record_path = confirmation_dir / "confirmed_selection.json"
    record_path.write_text(
        json.dumps(
            confirmation_record,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    confirmation_record["record_file"] = str(record_path)
    return confirmation_record


def load_confirmation_context(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Load a previously generated confirmation context."""
    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    path = confirmation_dir / "confirmation_context.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Confirmation context not found: {path}"
        )

    context = json.loads(path.read_text(encoding="utf-8"))
    context["context_file"] = str(path)
    return context
