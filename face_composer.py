"""Deterministic forensic face-composite builder for Third Eye."""

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image

from component_selector import (
    DEFAULT_CATALOG_PATH,
    load_catalog,
    select_category,
)
from face_schema import validate_analysis


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "static" / "outputs" / "cases"

DEFAULT_COMPONENTS = {
    "heads": "head_01",
    "necks": "neck_02",
}


def create_case_id():
    """Return a unique, filesystem-safe case identifier."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"case-{timestamp}-{suffix}"


def validate_case_id(case_id):
    """Reject unsafe case IDs instead of silently changing them."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError(
            "case_id may contain only letters, numbers, underscores and hyphens"
        )

    return case_id


def _catalog_entry_map(catalog):
    result = {}
    for category, entries in catalog["components"].items():
        for entry in entries:
            result[(category, entry["id"])] = entry
    return result


def _candidate_from_entry(entry, fallback=False):
    return {
        "id": entry["id"],
        "display_name": entry.get("display_name", entry["id"]),
        "file": entry["file"],
        "score": None if fallback else 0.0,
        "matched_attributes": {},
        "mismatched_attributes": {},
        "missing_attributes": [],
        "catalog_attributes": deepcopy(entry.get("attributes", {})),
        "aliases": list(entry.get("aliases", [])),
        "placement": deepcopy(entry.get("placement", {})),
        "review_status": entry.get("review_status", "unknown"),
        "fallback": fallback,
    }


def build_composition_plan(analysis, catalog=None):
    """Build an ordered component plan from structured description data."""
    errors = validate_analysis(analysis)
    if errors:
        raise ValueError("Invalid analysis: " + "; ".join(errors))

    if catalog is None:
        catalog = load_catalog()

    entry_map = _catalog_entry_map(catalog)
    plan = {category: [] for category in catalog["layer_order"]}
    warnings = []

    for category in catalog["layer_order"]:
        if category not in catalog["components"]:
            continue

        # Use a neutral head and neck when the witness did not describe them.
        if category in DEFAULT_COMPONENTS:
            result = select_category(
                analysis,
                category,
                top_k=1,
                catalog=catalog,
            )

            if result["candidates"] and result["candidates"][0]["score"] > 0:
                candidate = deepcopy(result["candidates"][0])
                candidate["fallback"] = False
                plan[category].append(candidate)
                continue

            default_id = DEFAULT_COMPONENTS[category]
            entry = entry_map.get((category, default_id))
            if entry is None:
                raise ValueError(
                    f"Default component {default_id} is missing from {category}"
                )

            plan[category].append(_candidate_from_entry(entry, fallback=True))
            warnings.append(
                f"No usable {category} description; used fallback {default_id}."
            )
            continue

        # Ears are individual left/right elements. Include all equally best
        # exact candidates so a size request can select the matching pair.
        top_k = 4 if category == "ears" else 1

        try:
            result = select_category(
                analysis,
                category,
                top_k=top_k,
                catalog=catalog,
            )
        except KeyError:
            continue

        if result["explicitly_absent"]:
            continue

        candidates = result["candidates"]
        if not candidates:
            continue

        top_score = candidates[0]["score"]
        if top_score <= 0:
            warnings.append(
                f"No {category} component matched the requested attributes."
            )
            continue

        if category == "ears":
            chosen = [
                deepcopy(candidate)
                for candidate in candidates
                if candidate["score"] == top_score
            ]
        else:
            chosen = [deepcopy(candidates[0])]

        for candidate in chosen:
            candidate["fallback"] = False

        plan[category].extend(chosen)

    return {
        "canvas": deepcopy(catalog["canvas"]),
        "layer_order": list(catalog["layer_order"]),
        "components": plan,
        "warnings": warnings,
    }


def _safe_component_path(component_root, relative_path):
    root = Path(component_root).resolve()
    candidate = (root / relative_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Unsafe component path outside asset root: {relative_path}"
        ) from error

    if not candidate.exists():
        raise FileNotFoundError(f"Component image not found: {candidate}")

    return candidate


def render_composition_plan(
    plan,
    output_path,
    component_root=None,
):
    """Render a composition plan to a transparent PNG on a white canvas."""
    if component_root is None:
        component_root = DEFAULT_CATALOG_PATH.parent

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_width = int(plan["canvas"]["width"])
    canvas_height = int(plan["canvas"]["height"])
    canvas = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (255, 255, 255, 255),
    )

    rendered = []

    for category in plan["layer_order"]:
        for component in plan["components"].get(category, []):
            placement = component["placement"]
            x = placement.get("x")
            y = placement.get("y")
            width = placement.get("width")
            height = placement.get("height")

            if None in (x, y, width, height):
                raise ValueError(
                    f"Incomplete placement for component {component['id']}"
                )

            image_path = _safe_component_path(
                component_root,
                component["file"],
            )

            with Image.open(image_path) as source_image:
                layer = source_image.convert("RGBA")

            target_size = (int(width), int(height))
            if layer.size != target_size:
                layer = layer.resize(target_size, Image.Resampling.LANCZOS)

            canvas.alpha_composite(layer, (int(x), int(y)))
            rendered.append(
                {
                    "category": category,
                    "id": component["id"],
                    "file": component["file"],
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "fallback": bool(component.get("fallback", False)),
                }
            )

    canvas.save(output_path, format="PNG")

    return {
        "output_path": str(output_path),
        "width": canvas_width,
        "height": canvas_height,
        "rendered_components": rendered,
    }


def compose_face(
    analysis,
    case_id=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Select, compose and save a per-case forensic reference image."""
    catalog = load_catalog(catalog_path)
    plan = build_composition_plan(analysis, catalog=catalog)

    if case_id is None:
        case_id = create_case_id()
    else:
        case_id = validate_case_id(case_id)

    output_root = Path(output_root)
    case_dir = output_root / case_id

    if case_dir.exists():
        raise FileExistsError(
            f"Case output directory already exists: {case_dir}"
        )

    case_dir.mkdir(parents=True, exist_ok=False)

    image_path = case_dir / "reference_composite.png"
    analysis_path = case_dir / "analysis.json"
    selection_path = case_dir / "selection.json"

    render_result = render_composition_plan(
        plan,
        image_path,
        component_root=Path(catalog_path).parent,
    )

    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "case_id": case_id,
        "case_directory": str(case_dir),
        "reference_image": str(image_path),
        "analysis_file": str(analysis_path),
        "selection_file": str(selection_path),
        "warnings": list(plan["warnings"]),
        "rendered_components": render_result["rendered_components"],
        "width": render_result["width"],
        "height": render_result["height"],
    }
