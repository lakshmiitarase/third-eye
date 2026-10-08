"""Backend for a witness-assisted manual forensic component editor."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from component_selector import (
    CATEGORY_RULES,
    load_catalog,
    select_category,
)
from confirmation_service import DEFAULT_CONFIRMATION_ROOT
from face_composer import (
    DEFAULT_CATALOG_PATH,
    render_composition_plan,
    validate_case_id,
)


EDITOR_CATEGORIES = [
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

NONE_ALLOWED_CATEGORIES = {
    "moustaches",
    "beards",
}

EDITOR_DIRECTORY_NAME = "manual_editor"
MIN_CONTRADICTION_REASON_LENGTH = 10
ABSENCE_VALUES = {"none", "absent", "no", False}


def _confirmation_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id


def _validate_confirmation_context(context):
    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary")

    required = {
        "confirmation_id",
        "description",
        "analysis",
        "plan",
    }
    missing = required - set(context)
    if missing:
        raise ValueError(
            f"Confirmation context is missing keys: {sorted(missing)}"
        )

    validate_case_id(context["confirmation_id"])


def _candidate_from_entry(entry):
    return {
        "id": entry["id"],
        "display_name": entry.get("display_name", entry["id"]),
        "file": entry["file"],
        "score": None,
        "matched_attributes": {},
        "mismatched_attributes": {},
        "missing_attributes": [],
        "catalog_attributes": deepcopy(entry.get("attributes", {})),
        "aliases": list(entry.get("aliases", [])),
        "placement": deepcopy(entry.get("placement", {})),
        "review_status": entry.get("review_status", "unknown"),
        "fallback": False,
    }


def _match_status(candidate, requested_attributes):
    if not requested_attributes:
        return "unrequested"

    if (
        candidate.get("score") == 1.0
        and not candidate.get("mismatched_attributes")
        and not candidate.get("missing_attributes")
    ):
        return "exact"

    if (candidate.get("score") or 0) > 0:
        return "partial"

    if candidate.get("mismatched_attributes"):
        return "contradictory"

    return "unmatched"


def _category_options(analysis, category, catalog):
    entries = [
        entry
        for entry in catalog["components"].get(category, [])
        if entry.get("enabled", True) is not False
    ]

    if not entries:
        return [], {}

    requested = {}
    for feature_name in CATEGORY_RULES[category]:
        value = analysis["features"][feature_name]["value"]
        if value is not None:
            requested[feature_name] = value

    if requested:
        selection = select_category(
            analysis,
            category,
            top_k=len(entries),
            catalog=catalog,
        )
        if selection["candidates"]:
            options = deepcopy(selection["candidates"])
        else:
            # Explicit absence still needs manual component choices visible.
            options = [_candidate_from_entry(entry) for entry in entries]
    else:
        options = [_candidate_from_entry(entry) for entry in entries]

    explicit_absence = (
        category in NONE_ALLOWED_CATEGORIES
        and bool(requested)
        and all(value in ABSENCE_VALUES for value in requested.values())
    )

    for option in options:
        if explicit_absence:
            option["match_status"] = "contradictory"
            for feature_name, requested_value in requested.items():
                option.setdefault("mismatched_attributes", {}).setdefault(
                    feature_name,
                    {
                        "requested": requested_value,
                        "candidate": option.get("catalog_attributes", {}).get(
                            feature_name,
                            "component present",
                        ),
                    },
                )
        else:
            option["match_status"] = _match_status(option, requested)

    status_order = {
        "exact": 0,
        "partial": 1,
        "unrequested": 2,
        "unmatched": 3,
        "contradictory": 4,
    }
    options.sort(
        key=lambda item: (
            status_order[item["match_status"]],
            -(item.get("score") or 0),
            item["id"],
        )
    )
    return options, requested


def _initial_choices(context):
    choices = {}
    components = context["plan"]["components"]

    for category in EDITOR_CATEGORIES:
        selected = components.get(category, [])
        if selected:
            choices[category] = selected[0]["id"]
        elif category in NONE_ALLOWED_CATEGORIES:
            choices[category] = "none"

    return choices


def _option_map(session, category):
    category_record = next(
        (
            item
            for item in session["categories"]
            if item["name"] == category
        ),
        None,
    )
    if category_record is None:
        return {}

    return {
        option["id"]: option
        for option in category_record["options"]
    }


def _apply_choices_to_plan(session, choices):
    if not isinstance(choices, dict):
        raise TypeError("choices must be a dictionary")

    merged_choices = deepcopy(session["current_choices"])
    plan = deepcopy(session["base_plan"])

    for category, component_id in choices.items():
        if category not in EDITOR_CATEGORIES:
            raise ValueError(f"Category is not editable: {category}")

        if component_id in (None, "", "none"):
            if category not in NONE_ALLOWED_CATEGORIES:
                raise ValueError(
                    f"The none option is not allowed for {category}"
                )
            plan["components"][category] = []
            merged_choices[category] = "none"
            continue

        allowed = _option_map(session, category)
        if component_id not in allowed:
            raise ValueError(
                f"Unknown component '{component_id}' for {category}"
            )

        selected = deepcopy(allowed[component_id])
        selected.pop("match_status", None)
        selected["fallback"] = False
        plan["components"][category] = [selected]
        merged_choices[category] = component_id

    return plan, merged_choices


def _selected_contradictions(session, merged_choices):
    contradictions = []

    for category, component_id in merged_choices.items():
        if component_id in (None, "", "none"):
            continue

        option = _option_map(session, category).get(component_id)
        if not option or option.get("match_status") != "contradictory":
            continue

        category_record = next(
            item
            for item in session["categories"]
            if item["name"] == category
        )
        contradictions.append(
            {
                "category": category,
                "category_label": category_record["label"],
                "component_id": component_id,
                "display_name": option.get("display_name", component_id),
                "mismatched_attributes": deepcopy(
                    option.get("mismatched_attributes", {})
                ),
            }
        )

    return contradictions


def _write_session(session, editor_dir):
    path = editor_dir / "editor_session.json"
    serializable = deepcopy(session)
    serializable.pop("session_file", None)
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    session["session_file"] = str(path)
    return path


def create_editor_session(
    context,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Create a manual editor session and render its initial preview."""
    _validate_confirmation_context(context)

    confirmation_id = context["confirmation_id"]
    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    editor_dir = confirmation_dir / EDITOR_DIRECTORY_NAME
    editor_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(catalog_path)
    categories = []

    for category in EDITOR_CATEGORIES:
        options, requested = _category_options(
            context["analysis"],
            category,
            catalog,
        )
        categories.append(
            {
                "name": category,
                "label": CATEGORY_LABELS[category],
                "requested_attributes": requested,
                "none_allowed": category in NONE_ALLOWED_CATEGORIES,
                "options": options,
            }
        )

    session = {
        "confirmation_id": confirmation_id,
        "description": context["description"],
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "analysis": deepcopy(context["analysis"]),
        "base_plan": deepcopy(context["plan"]),
        "current_choices": _initial_choices(context),
        "categories": categories,
        "preview_image": str(editor_dir / "editor_preview.png"),
        "finalized": False,
    }

    plan, current_choices = _apply_choices_to_plan(session, {})
    session["current_choices"] = current_choices
    render_composition_plan(
        plan,
        session["preview_image"],
        component_root=Path(catalog_path).parent,
    )
    _write_session(session, editor_dir)
    return session


def load_editor_session(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Load a previously created manual editor session."""
    editor_dir = (
        _confirmation_directory(confirmation_id, output_root)
        / EDITOR_DIRECTORY_NAME
    )
    path = editor_dir / "editor_session.json"

    if not path.is_file():
        raise FileNotFoundError(f"Editor session not found: {path}")

    session = json.loads(path.read_text(encoding="utf-8"))
    session["session_file"] = str(path)
    return session


def update_editor_preview(
    session,
    choices,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Apply validated manual choices and refresh the preview image."""
    confirmation_id = validate_case_id(session["confirmation_id"])
    editor_dir = (
        _confirmation_directory(confirmation_id, output_root)
        / EDITOR_DIRECTORY_NAME
    )
    editor_dir.mkdir(parents=True, exist_ok=True)

    plan, merged_choices = _apply_choices_to_plan(session, choices)
    preview_path = editor_dir / "editor_preview.png"
    render_result = render_composition_plan(
        plan,
        preview_path,
        component_root=Path(catalog_path).parent,
    )

    updated = deepcopy(session)
    updated["current_choices"] = merged_choices
    updated["preview_image"] = str(preview_path)
    updated["updated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    updated["preview_components"] = render_result[
        "rendered_components"
    ]
    _write_session(updated, editor_dir)
    return updated


def finalize_editor_selection(
    session,
    choices=None,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
    contradiction_reason=None,
):
    """Save the manually confirmed structural reference and audit record."""
    if choices is None:
        choices = {}

    confirmation_id = validate_case_id(session["confirmation_id"])
    editor_dir = (
        _confirmation_directory(confirmation_id, output_root)
        / EDITOR_DIRECTORY_NAME
    )
    editor_dir.mkdir(parents=True, exist_ok=True)

    plan, merged_choices = _apply_choices_to_plan(session, choices)
    contradictions = _selected_contradictions(session, merged_choices)
    reason = (
        contradiction_reason.strip()
        if isinstance(contradiction_reason, str)
        else ""
    )

    if contradictions and len(reason) < MIN_CONTRADICTION_REASON_LENGTH:
        raise ValueError(
            "A reason of at least "
            f"{MIN_CONTRADICTION_REASON_LENGTH} characters is required "
            "before saving contradictory choices."
        )

    if not contradictions:
        reason = None

    confirmed_path = editor_dir / "manual_confirmed_reference.png"
    render_result = render_composition_plan(
        plan,
        confirmed_path,
        component_root=Path(catalog_path).parent,
    )

    record = {
        "confirmation_id": confirmation_id,
        "finalized_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "mode": "witness_assisted_manual_component_editor",
        "description": session["description"],
        "analysis": deepcopy(session["analysis"]),
        "choices": merged_choices,
        "contradictions": contradictions,
        "contradiction_reason": reason,
        "plan": plan,
        "confirmed_image": str(confirmed_path),
        "rendered_components": render_result["rendered_components"],
        "notice": (
            "This structural composite was manually confirmed from "
            "catalogued facial components."
        ),
    }

    record_path = editor_dir / "manual_selection.json"
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record["record_file"] = str(record_path)

    finalized_session = deepcopy(session)
    finalized_session["current_choices"] = merged_choices
    finalized_session["finalized"] = True
    finalized_session["finalized_at"] = record["finalized_at"]
    _write_session(finalized_session, editor_dir)
    return record
