"""Tests for Phase 13B provisional geometry generation."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from component_geometry_manifest import (
    LANDMARK_NAMES,
    build_geometry_manifest,
    write_geometry_outputs,
)


def _entry(component_id="head_01", relative_file="heads/head_01.png"):
    return {
        "id": component_id,
        "display_name": "Head 01",
        "file": relative_file,
        "category": "heads",
        "source": {"project": "test", "category": "head", "original_file": "01.png"},
        "image": {"width": 20, "height": 30, "mode": "RGBA"},
        "placement": {
            "canvas_width": 620,
            "canvas_height": 620,
            "x": 300,
            "y": 200,
            "width": 20,
            "height": 30,
            "layer": "heads",
        },
        "attributes": {"face_shape": "oval", "jaw_shape": "rounded"},
        "aliases": ["oval face"],
        "review_status": "provisional",
        "enabled": True,
        "review_notes": "test",
    }


def _fixture(tmp_path, entries=None):
    root = tmp_path / "static" / "assets" / "face_components"
    root.mkdir(parents=True)
    entries = entries if entries is not None else [_entry()]
    catalog = {
        "schema_version": "1.0",
        "catalog_status": "provisional_semantic_labels",
        "canvas": {"width": 620, "height": 620, "background": "white"},
        "layer_order": ["heads", "glasses", "marks"],
        "components": {"heads": entries, "glasses": [], "marks": []},
    }
    path = root / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    for entry in entries:
        asset = root / entry["file"]
        asset.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGBA", (20, 30), (0, 0, 0, 0))
        for x in range(4, 16):
            for y in range(5, 26):
                image.putpixel((x, y), (80, 80, 80, 255))
        image.save(asset)
    return root, path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_builds_machine_measurements_for_every_component(tmp_path):
    _, catalog = _fixture(tmp_path)
    manifest = build_geometry_manifest(catalog)
    assert manifest["summary"]["component_count"] == 1
    record = manifest["components"][0]
    measurement = record["machine_measurements"]
    assert measurement["source_width_px"] == 20
    assert measurement["source_height_px"] == 30
    assert measurement["source_aspect_height_over_width"] == 1.5
    assert measurement["placement_center_x_px"] == 310
    assert measurement["placement_center_y_px"] == 215


def test_landmarks_are_blank_and_pending_expert_review(tmp_path):
    _, catalog = _fixture(tmp_path)
    record = build_geometry_manifest(catalog)["components"][0]
    assert set(record["expert_landmarks_canvas_px"]) == set(LANDMARK_NAMES)
    assert all(value == {"x": None, "y": None} for value in record["expert_landmarks_canvas_px"].values())
    assert record["geometry_review"]["status"] == "pending"


def test_alpha_bounds_and_fractions_are_measured(tmp_path):
    _, catalog = _fixture(tmp_path)
    measurement = build_geometry_manifest(catalog)["components"][0]["machine_measurements"]
    assert measurement["alpha_bbox_threshold_1_source_px"] == [4, 5, 16, 26]
    assert measurement["alpha_bbox_threshold_128_source_px"] == [4, 5, 16, 26]
    assert 0 < measurement["alpha_nonzero_fraction"] < 1
    assert measurement["alpha_nonzero_fraction"] == measurement["alpha_opaque_fraction"]


def test_source_hashes_match_and_sources_are_unchanged(tmp_path):
    root, catalog = _fixture(tmp_path)
    asset = root / "heads" / "head_01.png"
    before_catalog = _digest(catalog)
    before_asset = _digest(asset)
    manifest = build_geometry_manifest(catalog)
    assert _digest(catalog) == before_catalog
    assert _digest(asset) == before_asset
    assert manifest["summary"]["sources_unchanged"] is True
    assert manifest["components"][0]["source_sha256"] == before_asset


def test_semantic_attributes_remain_explicitly_provisional(tmp_path):
    _, catalog = _fixture(tmp_path)
    record = build_geometry_manifest(catalog)["components"][0]
    assert record["semantic_review_status"] == "provisional"
    assert record["provisional_semantic_attributes"]["face_shape"] == "oval"
    assert manifest_status(build_geometry_manifest(catalog)).startswith("provisional_")


def manifest_status(manifest):
    return manifest["status"]


def test_duplicate_component_id_is_rejected(tmp_path):
    first = _entry()
    second = _entry(relative_file="heads/head_02.png")
    _, catalog = _fixture(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Duplicate component ID"):
        build_geometry_manifest(catalog)


def test_unsafe_component_path_is_rejected(tmp_path):
    entry = _entry(relative_file="../../outside.png")
    _, catalog = _fixture(tmp_path, [entry])
    with pytest.raises(ValueError, match="escapes catalogue root"):
        build_geometry_manifest(catalog)


def test_missing_component_file_is_rejected(tmp_path):
    root, catalog = _fixture(tmp_path)
    (root / "heads" / "head_01.png").unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        build_geometry_manifest(catalog)


def test_metadata_size_mismatch_is_rejected(tmp_path):
    entry = _entry()
    entry["image"]["height"] = 31
    _, catalog = _fixture(tmp_path, [entry])
    with pytest.raises(ValueError, match="metadata mismatch"):
        build_geometry_manifest(catalog)


def test_placement_outside_canvas_is_rejected(tmp_path):
    entry = _entry()
    entry["placement"]["x"] = 610
    _, catalog = _fixture(tmp_path, [entry])
    with pytest.raises(ValueError, match="outside canvas"):
        build_geometry_manifest(catalog)


def test_writes_manifest_and_category_worksheets(tmp_path):
    _, catalog = _fixture(tmp_path)
    manifest = build_geometry_manifest(catalog)
    manifest_path = tmp_path / "out" / "geometry_manifest.provisional.json"
    sheets = tmp_path / "component_geometry_review_sheets"
    result = write_geometry_outputs(manifest, manifest_path, sheets)
    assert Path(result["manifest"]).is_file()
    assert len(result["worksheets"]) == 3
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["summary"]["component_count"] == 1
    with (sheets / "heads_geometry_review.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["geometry_review_status"] == "pending"
    assert rows[0]["left_eye_center_x"] == ""


def test_refuses_to_overwrite_manifest_or_review_directory(tmp_path):
    _, catalog = _fixture(tmp_path)
    manifest = build_geometry_manifest(catalog)
    manifest_path = tmp_path / "geometry_manifest.provisional.json"
    manifest_path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_geometry_outputs(manifest, manifest_path, tmp_path / "reviews")

    manifest_path.unlink()
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        write_geometry_outputs(manifest, manifest_path, reviews)
