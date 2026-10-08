"""Deterministic, non-generative cleanup for structural presentation copies.

The source manual_confirmed_reference.png is never modified. Cleanup preserves
canvas dimensions and pixel registration; it changes only grayscale tone,
near-white background values, light noise, and edge contrast.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from confirmation_service import DEFAULT_CONFIRMATION_ROOT
from face_composer import validate_case_id


PRESENTATION_DIRECTORY = "presentation"
OUTPUT_FILENAME = "cleaned_structural_presentation.png"
AUDIT_FILENAME = "cleaned_structural_presentation.json"

NOTICE = (
    "Non-generative presentation cleanup only. The source manually confirmed "
    "structural reference remains authoritative. No facial geometry, component "
    "selection, or witness attribute was intentionally changed."
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _manual_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id / "manual_editor"


def _mean_absolute_difference(first, second):
    difference = ImageChops.difference(
        first.convert("L"),
        second.convert("L"),
    )
    return round(ImageStat.Stat(difference).mean[0], 4)


def clean_structural_reference(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Create one audited, geometry-preserving structural presentation copy."""
    manual_dir = _manual_directory(confirmation_id, output_root)
    source_path = manual_dir / "manual_confirmed_reference.png"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Manual confirmed structural reference not found: {source_path}"
        )

    output_dir = manual_dir / PRESENTATION_DIRECTORY
    output_path = output_dir / OUTPUT_FILENAME
    audit_path = output_dir / AUDIT_FILENAME
    if output_path.exists() or audit_path.exists():
        raise FileExistsError(
            "Cleaned structural presentation already exists; preserve it and "
            "do not silently overwrite the audited derivative"
        )

    source_hash_before = _sha256(source_path)
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")

    original_size = source.size
    grayscale = ImageOps.grayscale(source)
    normalized = ImageOps.autocontrast(
        grayscale,
        cutoff=(0.5, 1.0),
    )
    lightly_denoised = normalized.filter(ImageFilter.MedianFilter(size=3))
    blended = Image.blend(normalized, lightly_denoised, 0.18)
    sharpened = blended.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=45,
            threshold=5,
        )
    )
    cleaned_gray = sharpened.point(
        lambda value: 255 if value >= 248 else value
    )
    cleaned = cleaned_gray.convert("RGB")

    if cleaned.size != original_size:
        raise RuntimeError("Cleanup changed canvas dimensions")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.save(output_path, format="PNG", optimize=True)

    source_hash_after = _sha256(source_path)
    if source_hash_after != source_hash_before:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Authoritative source changed during cleanup")

    audit = {
        "confirmation_id": confirmation_id,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "mode": "deterministic_non_generative_presentation_cleanup",
        "source": str(source_path),
        "source_sha256": source_hash_before,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "canvas": {
            "width": original_size[0],
            "height": original_size[1],
            "dimensions_preserved": True,
            "pixel_registration_preserved": True,
            "crop_applied": False,
            "resize_applied": False,
        },
        "operations": [
            "grayscale_conversion",
            "global_autocontrast_cutoff_0.5_1.0_percent",
            "median_filter_3px_blended_at_18_percent",
            "unsharp_mask_radius_1_percent_45_threshold_5",
            "near_white_background_threshold_248",
        ],
        "mean_absolute_grayscale_difference": (
            _mean_absolute_difference(source, cleaned)
        ),
        "ai_model_used": False,
        "facial_geometry_change_authorized": False,
        "authoritative_status": "presentation_copy_only",
        "notice": NOTICE,
    }
    _write_json(audit_path, audit)
    audit["audit_file"] = str(audit_path)
    return audit
