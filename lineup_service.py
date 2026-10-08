"""Create distinct, attribute-matched structural face lineups.

Lineup variants use alternate positive-scoring catalogue candidates. They do
not use random prompt-only identities. Explicitly absent features remain absent.
"""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from confirmation_service import (
    CONFIRMABLE_CATEGORIES,
    DEFAULT_CONFIRMATION_ROOT,
)
from face_composer import (
    DEFAULT_CATALOG_PATH,
    render_composition_plan,
    validate_case_id,
)


DEFAULT_VARIANT_COUNT = 3
LINEUP_DIRECTORY_NAME = "structural_lineup"


def _confirmation_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id


def _validate_context(context):
    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary")

    required = {
        "confirmation_id",
        "description",
        "analysis",
        "selections",
        "plan",
    }
    missing = required - set(context)
    if missing:
        raise ValueError(
            f"Confirmation context is missing keys: {sorted(missing)}"
        )

    validate_case_id(context["confirmation_id"])

    if not isinstance(context["selections"], dict):
        raise ValueError("context.selections must be a dictionary")
    if not isinstance(context["plan"], dict):
        raise ValueError("context.plan must be a dictionary")


def _eligible_candidates(selection):
    """Return only safe alternatives that do not contradict the request.

    Fully matching candidates may vary by rank. If no complete match exists,
    retain only the highest-ranked positive candidate for every variant rather
    than introducing a partial match with known mismatches.
    """
    positive = [
        candidate
        for candidate in selection.get("candidates", [])
        if candidate.get("score", 0) > 0
    ]

    exact = [
        candidate
        for candidate in positive
        if candidate.get("score") == 1.0
        and not candidate.get("mismatched_attributes")
        and not candidate.get("missing_attributes")
    ]

    if exact:
        return exact

    return positive[:1]


def _variant_plan(context, rank_index):
    plan = deepcopy(context["plan"])
    choices = {}

    for category in CONFIRMABLE_CATEGORIES:
        selection = context["selections"].get(category)
        if not isinstance(selection, dict):
            continue

        if selection.get("explicitly_absent"):
            plan["components"][category] = []
            choices[category] = "none"
            continue

        candidates = _eligible_candidates(selection)
        if not candidates:
            # Preserve neutral fallbacks and empty unmentioned categories.
            continue

        chosen_index = rank_index if rank_index < len(candidates) else 0
        chosen = deepcopy(candidates[chosen_index])
        chosen["fallback"] = False
        plan["components"][category] = [chosen]
        choices[category] = chosen["id"]

    return plan, choices


def _choice_signature(choices):
    return tuple(
        (category, choices.get(category))
        for category in CONFIRMABLE_CATEGORIES
    )


def create_structural_lineup(
    context,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
    variant_count=DEFAULT_VARIANT_COUNT,
):
    """Render distinct structural alternatives from offered candidates."""
    _validate_context(context)

    if not isinstance(variant_count, int) or isinstance(variant_count, bool):
        raise TypeError("variant_count must be an integer")
    if variant_count < 2 or variant_count > 3:
        raise ValueError("variant_count must be between 2 and 3")

    confirmation_id = context["confirmation_id"]
    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    lineup_dir = confirmation_dir / LINEUP_DIRECTORY_NAME
    lineup_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    signatures = set()

    for index in range(variant_count):
        variant_id = f"variant_{index + 1}"
        plan, choices = _variant_plan(context, index)
        signature = _choice_signature(choices)

        if signature in signatures:
            raise ValueError(
                "Not enough distinct positive-scoring component "
                "combinations to create the requested lineup"
            )
        signatures.add(signature)

        image_path = lineup_dir / f"{variant_id}_reference.png"
        render_result = render_composition_plan(
            plan,
            image_path,
            component_root=Path(catalog_path).parent,
        )

        variants.append(
            {
                "variant_id": variant_id,
                "rank": index + 1,
                "choices": choices,
                "plan": plan,
                "image": str(image_path),
                "rendered_components": render_result[
                    "rendered_components"
                ],
                "warnings": list(plan.get("warnings", [])),
            }
        )

    lineup_context = {
        "confirmation_id": confirmation_id,
        "description": context["description"],
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "analysis": context["analysis"],
        "mode": "attribute_matched_structural_lineup",
        "variant_count": variant_count,
        "variants": variants,
        "selected_variant": None,
        "notice": (
            "Each variant uses different positive-scoring catalogue "
            "components matched to the same witness description."
        ),
    }

    context_path = lineup_dir / "lineup_context.json"
    context_path.write_text(
        json.dumps(lineup_context, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lineup_context["context_file"] = str(context_path)
    return lineup_context


def load_lineup_context(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Load a previously generated structural lineup."""
    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    path = (
        confirmation_dir
        / LINEUP_DIRECTORY_NAME
        / "lineup_context.json"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Lineup context not found: {path}")

    context = json.loads(path.read_text(encoding="utf-8"))
    context["context_file"] = str(path)
    return context


def confirm_lineup_variant(
    lineup_context,
    variant_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Confirm one structural variant and render an immutable reference."""
    if not isinstance(lineup_context, dict):
        raise TypeError("lineup_context must be a dictionary")
    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError("variant_id must be a non-empty string")

    confirmation_id = validate_case_id(
        lineup_context.get("confirmation_id")
    )

    variants = lineup_context.get("variants", [])
    selected = next(
        (
            variant
            for variant in variants
            if variant.get("variant_id") == variant_id
        ),
        None,
    )

    if selected is None:
        raise ValueError(f"Unknown lineup variant: {variant_id}")

    confirmation_dir = _confirmation_directory(
        confirmation_id,
        output_root,
    )
    lineup_dir = confirmation_dir / LINEUP_DIRECTORY_NAME
    lineup_dir.mkdir(parents=True, exist_ok=True)

    confirmed_path = lineup_dir / "confirmed_lineup_reference.png"
    render_result = render_composition_plan(
        deepcopy(selected["plan"]),
        confirmed_path,
        component_root=Path(catalog_path).parent,
    )

    record = {
        "confirmation_id": confirmation_id,
        "selected_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "mode": "confirmed_attribute_matched_lineup_variant",
        "selected_variant": variant_id,
        "description": lineup_context["description"],
        "analysis": lineup_context["analysis"],
        "choices": deepcopy(selected["choices"]),
        "plan": deepcopy(selected["plan"]),
        "confirmed_image": str(confirmed_path),
        "rendered_components": render_result["rendered_components"],
        "notice": (
            "This structural reference was selected from deterministic "
            "description-matched component alternatives."
        ),
    }

    record_path = lineup_dir / "lineup_selection.json"
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record["record_file"] = str(record_path)
    return record
