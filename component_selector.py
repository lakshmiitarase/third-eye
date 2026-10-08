"""Deterministic facial-component selection using catalog.json."""

import json
from copy import deepcopy
from pathlib import Path

from face_schema import validate_analysis


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "static"
    / "assets"
    / "face_components"
    / "catalog.json"
)

# Higher weights mean that an attribute is more important when ranking.
CATEGORY_RULES = {
    "heads": {
        "face_shape": 4.0,
        "face_width": 2.0,
        "jaw_shape": 2.0,
        "chin_shape": 1.0,
    },
    "hair": {
        "hair_length": 3.0,
        "hair_texture": 3.0,
        "hairline": 1.0,
    },
    "eyes": {
        "eye_shape": 3.0,
        "eye_size": 2.0,
        "eye_spacing": 1.0,
    },
    "eyebrows": {
        "eyebrow_shape": 3.0,
        "eyebrow_thickness": 2.0,
        "eyebrow_spacing": 1.0,
    },
    "noses": {
        "nose_width": 3.0,
        "nose_length": 3.0,
        "nose_bridge": 2.0,
        "nose_tip": 2.0,
    },
    "lips": {
        "lip_thickness": 3.0,
        "mouth_width": 2.0,
    },
    "moustaches": {
        "moustache_style": 5.0,
    },
    "beards": {
        "beard_style": 5.0,
    },
    "ears": {
        "ear_size": 3.0,
    },
    "necks": {
        "neck_width": 3.0,
    },
    "glasses": {
        "glasses": 5.0,
    },
}

NONE_VALUES = {
    "beards": ("beard_style", "none"),
    "moustaches": ("moustache_style", "none"),
    "glasses": ("glasses", "none"),
}


def load_catalog(catalog_path=DEFAULT_CATALOG_PATH):
    """Load and minimally validate the facial-component catalogue."""
    path = Path(catalog_path)

    if not path.exists():
        raise FileNotFoundError(f"Component catalogue not found: {path}")

    catalog = json.loads(path.read_text(encoding="utf-8"))

    required_keys = {"schema_version", "canvas", "components"}
    missing_keys = required_keys - set(catalog.keys())
    if missing_keys:
        raise ValueError(f"Catalogue is missing keys: {sorted(missing_keys)}")

    if not isinstance(catalog["components"], dict):
        raise ValueError("catalog.components must be a dictionary")

    return catalog


def _requested_attributes(analysis, category):
    requested = {}

    for feature_name in CATEGORY_RULES[category]:
        value = analysis["features"][feature_name]["value"]
        if value is not None:
            requested[feature_name] = value

    return requested


def _score_entry(entry, requested, rules):
    """Score one entry from 0.0 to 1.0 against requested attributes."""
    attributes = entry.get("attributes", {})
    total_weight = sum(rules[name] for name in requested)

    if total_weight == 0:
        return {
            "score": 0.0,
            "matched_weight": 0.0,
            "total_weight": 0.0,
            "matched": {},
            "mismatched": {},
            "missing": [],
        }

    matched_weight = 0.0
    matched = {}
    mismatched = {}
    missing = []

    for feature_name, requested_value in requested.items():
        weight = rules[feature_name]

        if feature_name not in attributes:
            missing.append(feature_name)
            continue

        catalog_value = attributes[feature_name]

        if catalog_value == requested_value:
            matched_weight += weight
            matched[feature_name] = requested_value
        else:
            mismatched[feature_name] = {
                "requested": requested_value,
                "candidate": catalog_value,
            }

    score = matched_weight / total_weight

    return {
        "score": round(score, 6),
        "matched_weight": matched_weight,
        "total_weight": total_weight,
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
    }


def select_category(
    analysis,
    category,
    top_k=3,
    catalog=None,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Return the top deterministic candidates for one category."""
    if category not in CATEGORY_RULES:
        raise KeyError(f"Unknown component category: {category}")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    validation_errors = validate_analysis(analysis)
    if validation_errors:
        raise ValueError(
            "Invalid description analysis: " + "; ".join(validation_errors)
        )

    if catalog is None:
        catalog = load_catalog(catalog_path)

    requested = _requested_attributes(analysis, category)

    if category in NONE_VALUES:
        feature_name, none_value = NONE_VALUES[category]
        if requested.get(feature_name) == none_value:
            return {
                "category": category,
                "requested_attributes": requested,
                "explicitly_absent": True,
                "reason": f"{feature_name} was explicitly set to none",
                "candidates": [],
            }

    if not requested:
        return {
            "category": category,
            "requested_attributes": {},
            "explicitly_absent": False,
            "reason": "No relevant attributes were provided",
            "candidates": [],
        }

    entries = catalog["components"].get(category, [])
    candidates = []

    for entry in entries:
        if entry.get("enabled", True) is False:
            continue

        score_details = _score_entry(
            entry,
            requested,
            CATEGORY_RULES[category],
        )

        candidate = {
            "id": entry["id"],
            "display_name": entry.get("display_name", entry["id"]),
            "file": entry["file"],
            "score": score_details["score"],
            "matched_attributes": score_details["matched"],
            "mismatched_attributes": score_details["mismatched"],
            "missing_attributes": score_details["missing"],
            "catalog_attributes": deepcopy(entry.get("attributes", {})),
            "aliases": list(entry.get("aliases", [])),
            "placement": deepcopy(entry.get("placement", {})),
            "review_status": entry.get("review_status", "unknown"),
        }
        candidates.append(candidate)

    # Deterministic ordering: score, number of matches, then stable ID.
    candidates.sort(
        key=lambda item: (
            -item["score"],
            -len(item["matched_attributes"]),
            item["id"],
        )
    )

    return {
        "category": category,
        "requested_attributes": requested,
        "explicitly_absent": False,
        "reason": None if candidates else "No enabled catalogue entries",
        "candidates": candidates[:top_k],
    }


def select_all_components(
    analysis,
    top_k=3,
    catalog=None,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Return candidate selections for every supported component category."""
    if catalog is None:
        catalog = load_catalog(catalog_path)

    return {
        category: select_category(
            analysis,
            category,
            top_k=top_k,
            catalog=catalog,
        )
        for category in CATEGORY_RULES
    }


def select_best_components(
    analysis,
    catalog=None,
    catalog_path=DEFAULT_CATALOG_PATH,
):
    """Return only the highest-ranked component in each applicable category."""
    selections = select_all_components(
        analysis,
        top_k=1,
        catalog=catalog,
        catalog_path=catalog_path,
    )

    best = {}

    for category, result in selections.items():
        best[category] = (
            result["candidates"][0]
            if result["candidates"]
            else None
        )

    return best


def validate_selected_files(
    selections,
    component_root=None,
):
    """Return missing files referenced by selection results."""
    if component_root is None:
        component_root = DEFAULT_CATALOG_PATH.parent
    else:
        component_root = Path(component_root)

    missing = []

    for result in selections.values():
        for candidate in result.get("candidates", []):
            path = component_root / candidate["file"]
            if not path.exists():
                missing.append(str(path))

    return missing
