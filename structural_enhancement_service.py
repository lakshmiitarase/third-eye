"""Stronger deterministic enhancement for structural presentation copies.

The authoritative manual reference and the earlier cleanup are never modified.
This module preserves canvas dimensions and pixel registration and uses no AI.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from confirmation_service import DEFAULT_CONFIRMATION_ROOT
from face_composer import validate_case_id

OUTPUT_FILENAME = "enhanced_structural_presentation_v2.png"
AUDIT_FILENAME = "enhanced_structural_presentation_v2.json"
NOTICE = (
    "Stronger deterministic presentation enhancement only. The original "
    "manually confirmed structural reference remains authoritative. No crop, "
    "resize, component replacement, generative AI, or geometric transform was used."
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_absolute_difference(first, second):
    difference = ImageChops.difference(first.convert("L"), second.convert("L"))
    return round(ImageStat.Stat(difference).mean[0], 4)


def enhance_structural_reference_v2(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    confirmation_id = validate_case_id(confirmation_id)
    manual_dir = Path(output_root) / confirmation_id / "manual_editor"
    source_path = manual_dir / "manual_confirmed_reference.png"
    if not source_path.is_file():
        raise FileNotFoundError(f"Manual structural reference not found: {source_path}")

    output_dir = manual_dir / "presentation"
    output_path = output_dir / OUTPUT_FILENAME
    audit_path = output_dir / AUDIT_FILENAME
    if output_path.exists() or audit_path.exists():
        raise FileExistsError(
            "V2 structural presentation already exists; audited derivatives are not overwritten"
        )

    source_hash_before = _sha256(source_path)
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")

    original_size = source.size
    gray = ImageOps.grayscale(source)
    normalized = ImageOps.autocontrast(gray, cutoff=(1.0, 1.0))
    denoised = normalized.filter(ImageFilter.MedianFilter(size=3))
    softened_noise = Image.blend(normalized, denoised, 0.12)

    # Gamma below 1 brightens midtones without moving or synthesizing pixels.
    gamma = 0.88
    gamma_lut = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
    brightened = softened_noise.point(gamma_lut)
    contrasted = ImageEnhance.Contrast(brightened).enhance(1.12)
    sharpened = contrasted.filter(
        ImageFilter.UnsharpMask(radius=1.1, percent=70, threshold=4)
    )
    cleaned_gray = sharpened.point(lambda value: 255 if value >= 242 else value)
    enhanced = cleaned_gray.convert("RGB")

    if enhanced.size != original_size:
        raise RuntimeError("Enhancement changed canvas dimensions")

    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced.save(output_path, format="PNG", optimize=True)

    source_hash_after = _sha256(source_path)
    if source_hash_after != source_hash_before:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Authoritative source changed during enhancement")

    audit = {
        "confirmation_id": confirmation_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "deterministic_non_generative_structural_enhancement_v2",
        "profile": "stronger_visible_cleanup_v2",
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
            "geometric_transform_applied": False,
        },
        "operations": [
            "grayscale_conversion",
            "global_autocontrast_cutoff_1_1_percent",
            "median_filter_3px_blended_at_12_percent",
            "midtone_gamma_0.88",
            "contrast_factor_1.12",
            "unsharp_mask_radius_1.1_percent_70_threshold_4",
            "near_white_background_threshold_242",
        ],
        "mean_absolute_grayscale_difference": _mean_absolute_difference(source, enhanced),
        "ai_model_used": False,
        "facial_geometry_change_authorized": False,
        "authoritative_status": "presentation_copy_only_unapproved",
        "notice": NOTICE,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit["audit_file"] = str(audit_path)
    return audit
