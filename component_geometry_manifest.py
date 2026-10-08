"""Build a provisional, read-only geometry sidecar for Third Eye components.

Machine-measurable image and placement values are recorded. Facial landmarks
remain null until a forensic artist or domain expert reviews them. Source PNGs,
catalog.json, and semantic review CSVs are never modified.
"""

import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image

LANDMARK_NAMES = (
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "mouth_center",
    "chin_point",
    "left_jaw_point",
    "right_jaw_point",
    "forehead_hairline_point",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_asset_path(root, relative_path):
    root = Path(root).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Component path escapes catalogue root: {relative_path}"
        ) from error
    return candidate


def _threshold_bbox(alpha, threshold):
    mask = alpha.point(lambda value: 255 if value >= threshold else 0)
    bbox = mask.getbbox()
    return list(bbox) if bbox else None


def _placed_bbox(source_bbox, placement, source_size):
    if source_bbox is None:
        return None
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        return None
    scale_x = placement["width"] / source_width
    scale_y = placement["height"] / source_height
    left, top, right, bottom = source_bbox
    return [
        round(placement["x"] + left * scale_x, 3),
        round(placement["y"] + top * scale_y, 3),
        round(placement["x"] + right * scale_x, 3),
        round(placement["y"] + bottom * scale_y, 3),
    ]


def _snapshot_referenced_sources(catalog, catalogue_root, catalog_path):
    snapshot = {str(Path(catalog_path).resolve()): _sha256(catalog_path)}
    for entries in catalog.get("components", {}).values():
        for entry in entries:
            path = _safe_asset_path(catalogue_root, entry["file"])
            if path.is_file():
                snapshot[str(path)] = _sha256(path)
    return dict(sorted(snapshot.items()))


def build_geometry_manifest(catalog_path):
    """Build a provisional geometry manifest entirely in memory."""
    catalog_path = Path(catalog_path).resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalogue not found: {catalog_path}")

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError("catalog.json is not valid JSON") from error

    required = {"schema_version", "catalog_status", "canvas", "layer_order", "components"}
    missing = sorted(required - set(catalog))
    if missing:
        raise ValueError(f"Catalogue is missing fields: {missing}")
    if not isinstance(catalog["components"], dict):
        raise ValueError("catalog.components must be an object")

    canvas = catalog["canvas"]
    canvas_width = canvas.get("width")
    canvas_height = canvas.get("height")
    if not isinstance(canvas_width, int) or not isinstance(canvas_height, int) or canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Catalogue canvas dimensions must be positive integers")

    catalogue_root = catalog_path.parent
    source_before = _snapshot_referenced_sources(catalog, catalogue_root, catalog_path)
    identifiers = set()
    relative_files = set()
    records = []
    category_counts = {}

    for category, entries in sorted(catalog["components"].items()):
        if not isinstance(entries, list):
            raise ValueError(f"Category {category} must contain a list")
        category_counts[category] = len(entries)

        for entry in sorted(entries, key=lambda item: item.get("id", "")):
            component_id = entry.get("id")
            relative_file = entry.get("file")
            if not component_id or not isinstance(component_id, str):
                raise ValueError(f"Invalid component ID in category {category}")
            if component_id in identifiers:
                raise ValueError(f"Duplicate component ID: {component_id}")
            identifiers.add(component_id)
            if not relative_file or not isinstance(relative_file, str):
                raise ValueError(f"Invalid component file for {component_id}")
            normalized_file = relative_file.replace("\\", "/")
            if normalized_file in relative_files:
                raise ValueError(f"Duplicate component file: {normalized_file}")
            relative_files.add(normalized_file)
            if entry.get("category") != category:
                raise ValueError(f"Category mismatch for {component_id}")

            asset_path = _safe_asset_path(catalogue_root, relative_file)
            if not asset_path.is_file():
                raise FileNotFoundError(f"Component file not found: {asset_path}")

            placement = entry.get("placement")
            if not isinstance(placement, dict):
                raise ValueError(f"Missing placement for {component_id}")
            required_placement = ("x", "y", "width", "height", "canvas_width", "canvas_height", "layer")
            if any(name not in placement for name in required_placement):
                raise ValueError(f"Incomplete placement for {component_id}")
            values = [placement[name] for name in ("x", "y", "width", "height")]
            if not all(isinstance(value, int) for value in values):
                raise ValueError(f"Non-integer placement for {component_id}")
            x, y, placed_width, placed_height = values
            if placed_width <= 0 or placed_height <= 0:
                raise ValueError(f"Invalid placement size for {component_id}")
            if x < 0 or y < 0 or x + placed_width > canvas_width or y + placed_height > canvas_height:
                raise ValueError(f"Placement outside canvas for {component_id}")

            with Image.open(asset_path) as source:
                source.load()
                actual_mode = source.mode
                actual_size = source.size
                rgba = source.convert("RGBA")
                alpha = rgba.getchannel("A")
                alpha_min, alpha_max = alpha.getextrema()
                alpha_histogram = alpha.histogram()
                alpha_nonzero = sum(alpha_histogram[1:])
                alpha_opaque = alpha_histogram[255]
                pixel_count = actual_size[0] * actual_size[1]
                bbox_alpha_1 = _threshold_bbox(alpha, 1)
                bbox_alpha_128 = _threshold_bbox(alpha, 128)

            image_meta = entry.get("image", {})
            expected_size = (image_meta.get("width"), image_meta.get("height"))
            if expected_size != actual_size:
                raise ValueError(
                    f"Image metadata mismatch for {component_id}: actual {actual_size}, metadata {expected_size}"
                )
            if image_meta.get("mode") != actual_mode:
                raise ValueError(
                    f"Image mode mismatch for {component_id}: actual {actual_mode}, metadata {image_meta.get('mode')}"
                )

            landmarks = {name: {"x": None, "y": None} for name in LANDMARK_NAMES}
            record = {
                "id": component_id,
                "display_name": entry.get("display_name", component_id),
                "category": category,
                "file": normalized_file,
                "source_sha256": _sha256(asset_path),
                "semantic_review_status": entry.get("review_status", "unknown"),
                "provisional_semantic_attributes": deepcopy(entry.get("attributes", {})),
                "machine_measurements": {
                    "source_width_px": actual_size[0],
                    "source_height_px": actual_size[1],
                    "source_mode": actual_mode,
                    "source_aspect_height_over_width": round(actual_size[1] / actual_size[0], 6),
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "alpha_nonzero_fraction": round(alpha_nonzero / pixel_count, 6),
                    "alpha_opaque_fraction": round(alpha_opaque / pixel_count, 6),
                    "alpha_bbox_threshold_1_source_px": bbox_alpha_1,
                    "alpha_bbox_threshold_128_source_px": bbox_alpha_128,
                    "placement_x_px": x,
                    "placement_y_px": y,
                    "placement_width_px": placed_width,
                    "placement_height_px": placed_height,
                    "placement_aspect_height_over_width": round(placed_height / placed_width, 6),
                    "placement_center_x_px": round(x + placed_width / 2, 3),
                    "placement_center_y_px": round(y + placed_height / 2, 3),
                    "placement_center_x_canvas_fraction": round((x + placed_width / 2) / canvas_width, 6),
                    "placement_center_y_canvas_fraction": round((y + placed_height / 2) / canvas_height, 6),
                    "placement_width_canvas_fraction": round(placed_width / canvas_width, 6),
                    "placement_height_canvas_fraction": round(placed_height / canvas_height, 6),
                    "scale_x_from_source": round(placed_width / actual_size[0], 6),
                    "scale_y_from_source": round(placed_height / actual_size[1], 6),
                    "alpha_bbox_threshold_1_canvas_px": _placed_bbox(bbox_alpha_1, placement, actual_size),
                    "alpha_bbox_threshold_128_canvas_px": _placed_bbox(bbox_alpha_128, placement, actual_size),
                },
                "expert_landmarks_canvas_px": landmarks,
                "geometry_review": {
                    "status": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "notes": None,
                },
            }
            records.append(record)

    source_after = _snapshot_referenced_sources(catalog, catalogue_root, catalog_path)
    if source_before != source_after:
        raise RuntimeError("A source file changed during geometry measurement")

    return {
        "schema_version": "1.0",
        "status": "provisional_machine_measurements_expert_landmarks_pending",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "Third Eye Phase 13B",
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": _sha256(catalog_path),
        "source_catalog_schema_version": catalog.get("schema_version"),
        "source_catalog_status": catalog.get("catalog_status"),
        "canvas": deepcopy(canvas),
        "landmark_coordinate_system": {
            "space": "620x620 composition canvas",
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "units": "pixels",
            "unreviewed_value": None,
        },
        "landmark_definitions": {
            "left_eye_center": "Subject's anatomical left eye center",
            "right_eye_center": "Subject's anatomical right eye center",
            "nose_tip": "Visible center of nose tip",
            "mouth_center": "Center between lip corners",
            "chin_point": "Lowest central chin point",
            "left_jaw_point": "Subject's anatomical left jaw angle",
            "right_jaw_point": "Subject's anatomical right jaw angle",
            "forehead_hairline_point": "Central visible forehead or hairline reference",
        },
        "summary": {
            "component_count": len(records),
            "category_counts": category_counts,
            "machine_measured_component_count": len(records),
            "expert_reviewed_component_count": 0,
            "sources_unchanged": True,
        },
        "limitations": [
            "Machine measurements describe files and placements, not witness similarity.",
            "Semantic attributes remain provisional.",
            "Landmarks are intentionally null until expert review.",
            "This sidecar is not used by component selection in Phase 13B.",
        ],
        "components": sorted(records, key=lambda item: (item["category"], item["id"])),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
    }


def _worksheet_fieldnames():
    fields = [
        "id",
        "category",
        "file",
        "semantic_review_status",
        "semantic_attributes_json",
        "source_sha256",
        "source_width_px",
        "source_height_px",
        "source_aspect_height_over_width",
        "placement_x_px",
        "placement_y_px",
        "placement_width_px",
        "placement_height_px",
        "placement_aspect_height_over_width",
        "placement_center_x_px",
        "placement_center_y_px",
        "alpha_nonzero_fraction",
        "alpha_opaque_fraction",
    ]
    for landmark in LANDMARK_NAMES:
        fields.extend([f"{landmark}_x", f"{landmark}_y"])
    fields.extend(["geometry_review_status", "reviewer", "reviewed_at", "notes"])
    return fields


def write_geometry_outputs(manifest, manifest_path, worksheet_root):
    """Write a new sidecar and new expert worksheets; never overwrite."""
    manifest_path = Path(manifest_path)
    worksheet_root = Path(worksheet_root)
    if manifest_path.exists():
        raise FileExistsError(f"Geometry manifest already exists: {manifest_path}")
    if worksheet_root.exists():
        raise FileExistsError(f"Geometry review directory already exists: {worksheet_root}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    worksheet_root.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fields = _worksheet_fieldnames()
    grouped = {}
    for component in manifest["components"]:
        grouped.setdefault(component["category"], []).append(component)

    all_categories = sorted(manifest["summary"]["category_counts"])
    worksheet_paths = []
    for category in all_categories:
        path = worksheet_root / f"{category}_geometry_review.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for component in grouped.get(category, []):
                measurement = component["machine_measurements"]
                row = {
                    "id": component["id"],
                    "category": component["category"],
                    "file": component["file"],
                    "semantic_review_status": component["semantic_review_status"],
                    "semantic_attributes_json": json.dumps(component["provisional_semantic_attributes"], sort_keys=True),
                    "source_sha256": component["source_sha256"],
                    "geometry_review_status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                }
                for name in (
                    "source_width_px", "source_height_px", "source_aspect_height_over_width",
                    "placement_x_px", "placement_y_px", "placement_width_px", "placement_height_px",
                    "placement_aspect_height_over_width", "placement_center_x_px", "placement_center_y_px",
                    "alpha_nonzero_fraction", "alpha_opaque_fraction",
                ):
                    row[name] = measurement[name]
                for landmark in LANDMARK_NAMES:
                    row[f"{landmark}_x"] = ""
                    row[f"{landmark}_y"] = ""
                writer.writerow(row)
        worksheet_paths.append(str(path))

    return {"manifest": str(manifest_path), "worksheets": worksheet_paths}
