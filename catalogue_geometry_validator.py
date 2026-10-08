"""Read-only catalogue and placement-geometry validator for Third Eye.

This module never writes to the catalogue, component images, or review CSVs.
It emits audit reports only into a caller-selected output directory.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

EXPECTED_CANVAS = (620, 620)
EMPTY_CATEGORIES_ALLOWED = {"glasses", "marks"}
REQUIRED_COMPONENT_FIELDS = {
    "id",
    "display_name",
    "file",
    "category",
    "image",
    "placement",
    "attributes",
    "review_status",
    "enabled",
}
REQUIRED_PLACEMENT_FIELDS = {
    "canvas_width",
    "canvas_height",
    "x",
    "y",
    "width",
    "height",
    "layer",
}


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
            f"Asset path escapes catalogue root: {relative_path}"
        ) from error
    return candidate


def _issue(level, code, message, component_id=None, details=None):
    result = {"level": level, "code": code, "message": message}
    if component_id is not None:
        result["component_id"] = component_id
    if details is not None:
        result["details"] = details
    return result


def _snapshot_sources(catalogue_root, review_root=None):
    paths = [p for p in Path(catalogue_root).rglob("*") if p.is_file()]
    if review_root and Path(review_root).is_dir():
        paths.extend(p for p in Path(review_root).rglob("*") if p.is_file())
    return {
        str(path.resolve()): _sha256(path)
        for path in sorted(set(paths), key=lambda item: str(item).lower())
    }


def _read_catalog(catalog_path):
    path = Path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalogue not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Catalogue is not valid JSON: {path}") from error


def _review_rows(review_root):
    rows = []
    if not review_root or not Path(review_root).is_dir():
        return rows
    for path in sorted(Path(review_root).glob("*_review.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), start=2):
                item = dict(row)
                item["_review_file"] = path.name
                item["_review_line"] = number
                rows.append(item)
    return rows


def audit_catalogue(catalog_path, review_root=None):
    """Return a complete read-only audit dictionary."""
    catalog_path = Path(catalog_path).resolve()
    catalogue_root = catalog_path.parent
    review_root = Path(review_root).resolve() if review_root else None
    source_before = _snapshot_sources(catalogue_root, review_root)
    catalog = _read_catalog(catalog_path)
    issues = []
    component_rows = []

    required_top = {"schema_version", "catalog_status", "canvas", "layer_order", "components"}
    missing_top = sorted(required_top - set(catalog))
    if missing_top:
        issues.append(_issue("error", "catalog.missing_top_level_fields", "Catalogue is missing required top-level fields.", details=missing_top))

    canvas = catalog.get("canvas", {})
    canvas_size = (canvas.get("width"), canvas.get("height"))
    if canvas_size != EXPECTED_CANVAS:
        issues.append(_issue("error", "catalog.unexpected_canvas", "Catalogue canvas must be 620 x 620.", details={"actual": canvas_size, "expected": EXPECTED_CANVAS}))

    components = catalog.get("components")
    if not isinstance(components, dict):
        issues.append(_issue("error", "catalog.invalid_components", "catalog.components must be an object."))
        components = {}

    layer_order = catalog.get("layer_order", [])
    if not isinstance(layer_order, list) or len(layer_order) != len(set(layer_order)):
        issues.append(_issue("error", "catalog.invalid_layer_order", "layer_order must be a unique list."))
        layer_order = []

    missing_layer_categories = sorted(set(components) - set(layer_order))
    unknown_layer_categories = sorted(set(layer_order) - set(components))
    if missing_layer_categories:
        issues.append(_issue("error", "catalog.categories_missing_from_layer_order", "Component categories are missing from layer_order.", details=missing_layer_categories))
    if unknown_layer_categories:
        issues.append(_issue("warning", "catalog.empty_or_unknown_layer_categories", "layer_order contains categories not represented in components.", details=unknown_layer_categories))

    if catalog.get("catalog_status") == "provisional_semantic_labels":
        issues.append(_issue("warning", "catalog.provisional_semantic_labels", "Semantic component labels remain provisional and require domain-expert review."))

    # The current schema has semantic labels and placements but no canonical
    # facial landmarks or measured morphology for geometry-aware selection.
    if "geometry_schema" not in catalog and "landmarks" not in catalog:
        issues.append(_issue("warning", "catalog.geometry_schema_missing", "No canonical landmark or measured face-geometry schema is defined; short-vs-long or proportion-aware matching cannot be validated."))

    id_locations = defaultdict(list)
    file_locations = defaultdict(list)
    referenced_files = set()
    review_statuses = Counter()

    for category, entries in sorted(components.items()):
        if not isinstance(entries, list):
            issues.append(_issue("error", "category.entries_not_list", f"Category {category} must contain a list."))
            continue
        if not entries and category not in EMPTY_CATEGORIES_ALLOWED:
            issues.append(_issue("warning", "category.unexpected_empty", f"Category {category} has no components."))

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                issues.append(_issue("error", "component.not_object", f"Entry {index} in {category} is not an object."))
                continue

            component_id = entry.get("id", f"{category}[{index}]")
            missing_fields = sorted(REQUIRED_COMPONENT_FIELDS - set(entry))
            if missing_fields:
                issues.append(_issue("error", "component.missing_fields", "Component is missing required fields.", component_id, missing_fields))

            id_locations[component_id].append(category)
            relative_file = entry.get("file")
            if isinstance(relative_file, str):
                file_locations[relative_file].append(component_id)
                referenced_files.add(relative_file.replace("\\", "/"))
            else:
                issues.append(_issue("error", "component.invalid_file", "Component file must be a relative string.", component_id))
                continue

            if entry.get("category") != category:
                issues.append(_issue("error", "component.category_mismatch", "Entry category differs from its catalogue section.", component_id, {"section": category, "entry": entry.get("category")}))

            placement = entry.get("placement", {})
            missing_placement = sorted(REQUIRED_PLACEMENT_FIELDS - set(placement)) if isinstance(placement, dict) else sorted(REQUIRED_PLACEMENT_FIELDS)
            if missing_placement:
                issues.append(_issue("error", "component.incomplete_placement", "Placement is incomplete.", component_id, missing_placement))
                placement = placement if isinstance(placement, dict) else {}

            if placement.get("layer") != category:
                issues.append(_issue("error", "component.layer_mismatch", "Placement layer differs from component category.", component_id, {"category": category, "layer": placement.get("layer")}))

            review_status = entry.get("review_status", "missing")
            review_statuses[review_status] += 1

            try:
                asset_path = _safe_asset_path(catalogue_root, relative_file)
            except ValueError as error:
                issues.append(_issue("error", "component.unsafe_path", str(error), component_id))
                continue

            row = {
                "id": component_id,
                "category": category,
                "file": relative_file.replace("\\", "/"),
                "enabled": entry.get("enabled"),
                "review_status": review_status,
                "exists": asset_path.is_file(),
                "sha256": None,
                "actual_width": None,
                "actual_height": None,
                "mode": None,
                "alpha_min": None,
                "alpha_max": None,
                "alpha_nonzero_fraction": None,
                "placement_x": placement.get("x"),
                "placement_y": placement.get("y"),
                "placement_width": placement.get("width"),
                "placement_height": placement.get("height"),
                "placement_aspect_height_over_width": None,
                "placement_center_x": None,
                "placement_center_y": None,
                "within_canvas": False,
                "attributes_json": json.dumps(entry.get("attributes", {}), sort_keys=True),
            }

            if not asset_path.is_file():
                issues.append(_issue("error", "component.file_missing", "Referenced component file does not exist.", component_id, relative_file))
                component_rows.append(row)
                continue

            row["sha256"] = _sha256(asset_path)
            try:
                with Image.open(asset_path) as image:
                    image.load()
                    actual_size = image.size
                    actual_mode = image.mode
                    rgba = image.convert("RGBA")
                    alpha = rgba.getchannel("A")
                    alpha_min, alpha_max = alpha.getextrema()
                    histogram = alpha.histogram()
                    nonzero = sum(histogram[1:])
                    total_pixels = actual_size[0] * actual_size[1]
            except Exception as error:
                issues.append(_issue("error", "component.image_unreadable", f"Pillow could not read the image: {error}", component_id))
                component_rows.append(row)
                continue

            row.update({
                "actual_width": actual_size[0],
                "actual_height": actual_size[1],
                "mode": actual_mode,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "alpha_nonzero_fraction": round(nonzero / total_pixels, 6) if total_pixels else 0.0,
            })

            if asset_path.suffix.lower() != ".png":
                issues.append(_issue("error", "component.not_png", "Component asset must be PNG.", component_id, asset_path.suffix))
            if actual_mode != "RGBA":
                issues.append(_issue("error", "component.not_rgba", "Component asset must use RGBA mode.", component_id, actual_mode))
            if alpha_min == 255:
                issues.append(_issue("error", "component.no_transparency", "Component has no transparent pixels.", component_id))
            if alpha_max == 0:
                issues.append(_issue("error", "component.fully_transparent", "Component is fully transparent.", component_id))

            image_meta = entry.get("image", {}) if isinstance(entry.get("image"), dict) else {}
            metadata_size = (image_meta.get("width"), image_meta.get("height"))
            if metadata_size != actual_size:
                issues.append(_issue("error", "component.image_metadata_size_mismatch", "Image dimensions differ from catalogue image metadata.", component_id, {"actual": actual_size, "metadata": metadata_size}))
            if image_meta.get("mode") != actual_mode:
                issues.append(_issue("error", "component.image_metadata_mode_mismatch", "Image mode differs from catalogue metadata.", component_id, {"actual": actual_mode, "metadata": image_meta.get("mode")}))

            placement_size = (placement.get("width"), placement.get("height"))
            if placement_size != actual_size:
                issues.append(_issue("warning", "component.render_resize_required", "Placement size differs from source size; renderer will resample the component.", component_id, {"source": actual_size, "placement": placement_size}))

            numeric = [placement.get(key) for key in ("x", "y", "width", "height")]
            if all(isinstance(value, int) for value in numeric):
                x, y, width, height = numeric
                within = x >= 0 and y >= 0 and width > 0 and height > 0 and x + width <= canvas.get("width", 0) and y + height <= canvas.get("height", 0)
                row["within_canvas"] = within
                if width > 0:
                    row["placement_aspect_height_over_width"] = round(height / width, 6)
                row["placement_center_x"] = round(x + width / 2, 3)
                row["placement_center_y"] = round(y + height / 2, 3)
                if not within:
                    issues.append(_issue("error", "component.placement_outside_canvas", "Placement is invalid or extends beyond the canvas.", component_id, {"x": x, "y": y, "width": width, "height": height, "canvas": canvas_size}))
            else:
                issues.append(_issue("error", "component.non_integer_placement", "Placement x, y, width, and height must be integers.", component_id, numeric))

            component_rows.append(row)

    for component_id, locations in id_locations.items():
        if len(locations) > 1:
            issues.append(_issue("error", "catalog.duplicate_component_id", "Component ID appears more than once.", component_id, locations))
    for relative_file, component_ids in file_locations.items():
        if len(component_ids) > 1:
            issues.append(_issue("error", "catalog.duplicate_component_file", "Component file is referenced more than once.", details={"file": relative_file, "component_ids": component_ids}))

    binary_groups = defaultdict(list)
    for row in component_rows:
        if row["sha256"]:
            binary_groups[row["sha256"]].append(row["file"])
    for digest, files in binary_groups.items():
        if len(files) > 1:
            issues.append(_issue("warning", "catalog.duplicate_binary_assets", "Multiple components have identical file bytes.", details={"sha256": digest, "files": sorted(files)}))

    actual_pngs = {
        path.relative_to(catalogue_root).as_posix()
        for path in catalogue_root.rglob("*.png")
        if path.is_file()
    }
    for relative_file in sorted(actual_pngs - referenced_files):
        issues.append(_issue("warning", "catalog.unreferenced_png", "PNG exists under the catalogue root but is not referenced by catalog.json.", details=relative_file))

    review_rows = _review_rows(review_root)
    review_by_id = defaultdict(list)
    for row in review_rows:
        review_by_id[row.get("id", "")].append(row)
    catalog_by_id = {row["id"]: row for row in component_rows}

    if review_root and not review_rows and component_rows:
        issues.append(_issue("warning", "reviews.no_rows", "No component review rows were found."))

    for component_id, component in catalog_by_id.items():
        rows = review_by_id.get(component_id, [])
        if not rows:
            issues.append(_issue("warning", "reviews.component_missing", "Catalogue component is absent from review CSVs.", component_id))
            continue
        if len(rows) > 1:
            issues.append(_issue("error", "reviews.duplicate_component", "Component appears more than once in review CSVs.", component_id, [{"file": row["_review_file"], "line": row["_review_line"]} for row in rows]))
            continue
        review = rows[0]
        for field in ("category", "file", "review_status"):
            review_value = review.get(field, "").replace("\\", "/")
            catalog_value = str(component[field]).replace("\\", "/")
            if review_value != catalog_value:
                issues.append(_issue("error", f"reviews.{field}_mismatch", f"Review CSV {field} differs from catalog.json.", component_id, {"catalog": catalog_value, "review": review_value, "review_file": review["_review_file"], "line": review["_review_line"]}))

    for component_id, rows in review_by_id.items():
        if component_id and component_id not in catalog_by_id:
            issues.append(_issue("warning", "reviews.unknown_component", "Review CSV contains an ID absent from catalog.json.", component_id, [{"file": row["_review_file"], "line": row["_review_line"]} for row in rows]))

    if review_statuses and set(review_statuses) == {"provisional"}:
        issues.append(_issue("warning", "reviews.all_components_provisional", "All catalogue components remain provisional; no domain-expert-reviewed component is recorded."))

    source_after = _snapshot_sources(catalogue_root, review_root)
    sources_unchanged = source_before == source_after
    if not sources_unchanged:
        issues.append(_issue("error", "audit.sources_changed_during_validation", "One or more source files changed during the read-only audit."))

    issue_counts = Counter(item["level"] for item in issues)
    category_counts = {category: len(entries) if isinstance(entries, list) else 0 for category, entries in sorted(components.items())}
    report = {
        "audit_version": "phase13a-1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "read_only_catalogue_and_geometry_validation",
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "catalog_schema_version": catalog.get("schema_version"),
        "catalog_status": catalog.get("catalog_status"),
        "canvas": canvas,
        "layer_order": layer_order,
        "summary": {
            "component_count": len(component_rows),
            "category_counts": category_counts,
            "actual_png_count": len(actual_pngs),
            "referenced_png_count": len(referenced_files),
            "review_row_count": len(review_rows),
            "review_status_counts": dict(sorted(review_statuses.items())),
            "error_count": issue_counts.get("error", 0),
            "warning_count": issue_counts.get("warning", 0),
            "sources_unchanged": sources_unchanged,
        },
        "limitations": [
            "Semantic labels are not geometry measurements.",
            "No canonical facial landmarks are present in schema version 1.0.",
            "Placement rectangles can be validated, but witness similarity cannot be inferred.",
            "Provisional labels require forensic-artist or domain-expert review.",
        ],
        "issues": sorted(issues, key=lambda item: (item["level"], item["code"], item.get("component_id", ""))),
        "components": sorted(component_rows, key=lambda row: (row["category"], row["id"])),
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
    }
    return report


def write_audit_reports(report, output_dir):
    """Write JSON, CSV, and text summaries outside the source catalogue."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase13a_catalogue_audit.json"
    csv_path = output_dir / "phase13a_component_geometry.csv"
    text_path = output_dir / "phase13a_summary.txt"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "id", "category", "file", "enabled", "review_status", "exists", "sha256",
        "actual_width", "actual_height", "mode", "alpha_min", "alpha_max",
        "alpha_nonzero_fraction", "placement_x", "placement_y", "placement_width",
        "placement_height", "placement_aspect_height_over_width", "placement_center_x",
        "placement_center_y", "within_canvas", "attributes_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["components"])

    summary = report["summary"]
    lines = [
        "THIRD EYE PHASE 13A — READ-ONLY CATALOGUE AUDIT",
        "=================================================",
        "",
        f"Created: {report['created_at']}",
        f"Catalogue: {report['catalog_path']}",
        f"Catalogue SHA-256: {report['catalog_sha256']}",
        f"Schema: {report['catalog_schema_version']}",
        f"Status: {report['catalog_status']}",
        f"Components: {summary['component_count']}",
        f"Referenced PNGs: {summary['referenced_png_count']}",
        f"Actual PNGs: {summary['actual_png_count']}",
        f"Review rows: {summary['review_row_count']}",
        f"Errors: {summary['error_count']}",
        f"Warnings: {summary['warning_count']}",
        f"Sources unchanged: {summary['sources_unchanged']}",
        "",
        "Category counts:",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "Issues:"])
    if report["issues"]:
        for item in report["issues"]:
            suffix = f" [{item['component_id']}]" if item.get("component_id") else ""
            lines.append(f"- {item['level'].upper()} {item['code']}{suffix}: {item['message']}")
    else:
        lines.append("- None")
    lines.extend(["", "Limitations:"])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "summary": str(text_path)}
