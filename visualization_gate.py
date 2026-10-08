"""Local-only gate for optional AI visualizations of manual references.

The manually confirmed structural composite remains authoritative. Generated
images start in pending_review and cannot become accepted unless every required
attribute and structural-reference check is explicitly passed by a human
reviewer. Importing this module does not load PyTorch or a diffusion model.
"""

import hashlib
import json
import re
import secrets
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image

from confirmation_service import DEFAULT_CONFIRMATION_ROOT
from face_composer import validate_case_id


VISUALIZATION_DIRECTORY_NAME = "visualizations"
CANDIDATE_ID_PATTERN = re.compile(r"^candidate-[0-9]{3}$")
STATUS_PENDING = "pending_review"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

GENERATION_PROFILES = {
    "ultra_locked": {
        "steps": 36,
        "strength": 0.30,
        "guidance_scale": 5.5,
        "control_scale": 1.20,
        "purpose": (
            "Terminal experiment with maximum structural lock; no retries "
            "after one reviewed candidate"
        ),
        "experimental": True,
        "maximum_candidates_per_confirmation": 1,
    },
    "conservative": {
        "steps": 30,
        "strength": 0.42,
        "guidance_scale": 6.0,
        "control_scale": 1.05,
        "purpose": "Maximum structural preservation on low-memory hardware",
    },
    "balanced": {
        "steps": 24,
        "strength": 0.60,
        "guidance_scale": 6.5,
        "control_scale": 0.75,
        "purpose": "More photorealistic freedom with higher drift risk",
    },
}

NOTICE = (
    "Optional AI visualization of a witness-assisted structural composite. "
    "It is not an exact identification and must not replace the manually "
    "confirmed structural reference."
)

FEATURE_LABELS = {
    "gender": "Gender presentation",
    "age_group": "Age group",
    "face_shape": "Face shape",
    "face_width": "Face width",
    "jaw_shape": "Jaw shape",
    "chin_shape": "Chin shape",
    "hair_length": "Hair length",
    "hair_texture": "Hair texture",
    "hair_colour": "Hair colour",
    "eye_shape": "Eye shape",
    "eye_size": "Eye size",
    "eyebrow_shape": "Eyebrow shape",
    "eyebrow_thickness": "Eyebrow thickness",
    "nose_width": "Nose width",
    "nose_length": "Nose length",
    "lip_thickness": "Lip thickness",
    "moustache_style": "Moustache",
    "beard_style": "Beard",
    "glasses": "Glasses",
    "skin_tone": "Skin tone",
    "expression": "Expression",
    "pose": "Pose",
}

COMPONENT_LABELS = {
    "heads": "Face-shape component",
    "hair": "Hair component",
    "eyes": "Eye component",
    "eyebrows": "Eyebrow component",
    "noses": "Nose component",
    "lips": "Lip component",
    "moustaches": "Moustache component",
    "beards": "Beard component",
}


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path, payload):
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id / "manual_editor"


def _visualization_root(confirmation_id, output_root):
    return (
        _manual_directory(confirmation_id, output_root)
        / VISUALIZATION_DIRECTORY_NAME
    )


def _validate_candidate_id(candidate_id):
    if not isinstance(candidate_id, str) or not CANDIDATE_ID_PATTERN.fullmatch(
        candidate_id
    ):
        raise ValueError("candidate_id must use the form candidate-001")
    return candidate_id


def load_authoritative_manual_record(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Load and validate the authoritative manual selection and image."""
    manual_dir = _manual_directory(confirmation_id, output_root)
    selection_path = manual_dir / "manual_selection.json"
    record = _read_json(selection_path)

    if record.get("confirmation_id") != confirmation_id:
        raise ValueError("Manual selection confirmation_id does not match")
    if record.get("mode") != "witness_assisted_manual_component_editor":
        raise ValueError("Manual selection has an unsupported mode")

    reference_path = manual_dir / "manual_confirmed_reference.png"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Manual confirmed reference not found: {reference_path}"
        )

    record["record_file"] = str(selection_path)
    record["confirmed_image"] = str(reference_path)
    return record


def _normalized_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("_", " ").strip()


def _controlled_features(analysis):
    features = analysis.get("features", {}) if isinstance(analysis, dict) else {}
    controlled = {}
    for name, state in features.items():
        if not isinstance(state, dict):
            continue
        value = state.get("value")
        if value in (None, "", "unknown"):
            continue
        controlled[name] = {
            "value": value,
            "negated": bool(state.get("negated")),
            "source_text": state.get("source_text", ""),
        }
    return controlled


def _skin_tone_is_explicit(analysis):
    state = (
        analysis.get("features", {}).get("skin_tone")
        if isinstance(analysis, dict)
        else None
    )
    return bool(
        isinstance(state, dict)
        and state.get("value") not in (None, "", "unknown")
        and not state.get("negated")
    )


def build_review_checks(manual_record):
    """Build the mandatory human-review checklist for one candidate."""
    checks = [
        {
            "id": "structural_reference_match",
            "label": "Overall geometry matches the authoritative structural reference",
            "expected": "manual_confirmed_reference.png",
            "source": "mandatory_gate",
        },
        {
            "id": "front_facing_neutral_single_face",
            "label": "Single, front-facing face with neutral expression",
            "expected": "yes",
            "source": "mandatory_gate",
        },
        {
            "id": "no_unrequested_features",
            "label": "No unrequested facial hair, glasses, or other salient features",
            "expected": "yes",
            "source": "mandatory_gate",
        },
    ]

    for name, state in sorted(
        _controlled_features(manual_record.get("analysis", {})).items()
    ):
        expected = _normalized_value(state["value"])
        checks.append(
            {
                "id": f"feature_{name}",
                "label": FEATURE_LABELS.get(
                    name,
                    name.replace("_", " ").title(),
                ),
                "expected": expected,
                "negated": state["negated"],
                "source": "witness_description",
            }
        )

    for category, component_id in sorted(
        manual_record.get("choices", {}).items()
    ):
        if category not in COMPONENT_LABELS:
            continue
        checks.append(
            {
                "id": f"component_{category}",
                "label": COMPONENT_LABELS[category],
                "expected": component_id,
                "source": "manual_structural_selection",
            }
        )

    checks.append(
        {
            "id": "color_policy",
            "label": "Output follows the explicit skin-tone colour policy",
            "expected": (
                "color permitted"
                if _skin_tone_is_explicit(manual_record.get("analysis", {}))
                else "photorealistic monochrome"
            ),
            "source": "system_policy",
        }
    )
    return checks


def _next_candidate_id(root):
    root = Path(root)
    existing = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and CANDIDATE_ID_PATTERN.fullmatch(child.name):
                existing.append(int(child.name.split("-")[1]))
    number = max(existing, default=0) + 1
    if number > 999:
        raise RuntimeError("Candidate limit reached")
    return f"candidate-{number:03d}"


def _enforce_profile_candidate_limit(root, profile_name, profile_settings):
    maximum = profile_settings.get("maximum_candidates_per_confirmation")
    if maximum is None:
        return

    used = 0
    root = Path(root)
    if root.is_dir():
        for candidate_dir in root.iterdir():
            candidate_path = candidate_dir / "candidate.json"
            if not candidate_path.is_file():
                continue
            try:
                candidate = _read_json(candidate_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            generation_profile = candidate.get("generation_profile", {})
            if generation_profile.get("name") == profile_name:
                used += 1

    if used >= maximum:
        raise ValueError(
            f"Profile {profile_name} permits only {maximum} candidate per "
            "confirmation; stop diffusion tuning and retain the structural "
            "reference if that candidate fails review."
        )


def get_generation_profile(profile_name):
    if not isinstance(profile_name, str):
        raise TypeError("profile must be a string")
    if profile_name not in GENERATION_PROFILES:
        raise ValueError(
            f"Unknown generation profile: {profile_name}; "
            f"choose from {sorted(GENERATION_PROFILES)}"
        )
    return deepcopy(GENERATION_PROFILES[profile_name])


def _default_generate(reference_path, analysis, output_dir, **settings):
    # Lazy import: importing visualization_gate never initializes AI libraries.
    from photorealistic_generator import generate_photorealistic

    return generate_photorealistic(
        reference_path,
        analysis,
        output_dir,
        **settings,
    )


def _move_generated_file(source, destination, label):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Generated {label} not found: {source}")
    destination = Path(destination)
    source.replace(destination)
    return destination


def _enforce_color_policy(portrait_path, analysis):
    explicit_skin_tone = _skin_tone_is_explicit(analysis)
    if explicit_skin_tone:
        return {
            "mode": "color_permitted_explicit_skin_tone",
            "postprocessed": False,
        }

    path = Path(portrait_path)
    with Image.open(path) as image:
        monochrome = image.convert("L").convert("RGB")
        monochrome.save(path)
    return {
        "mode": "monochrome_unspecified_skin_tone",
        "postprocessed": True,
    }


def generate_pending_visualization(
    confirmation_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    generation_function=None,
    generation_settings=None,
    private_seed=None,
    profile="conservative",
):
    """Generate one private-seed candidate in pending-review state."""
    manual = load_authoritative_manual_record(
        confirmation_id,
        output_root=output_root,
    )
    profile_settings = get_generation_profile(profile)
    if generation_settings is not None and not isinstance(
        generation_settings,
        dict,
    ):
        raise TypeError("generation_settings must be a dictionary")

    seed = private_seed
    if seed is None:
        seed = secrets.randbelow(2**32)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("private_seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("private_seed must be between 0 and 4294967295")

    visual_root = _visualization_root(confirmation_id, output_root)
    _enforce_profile_candidate_limit(
        visual_root,
        profile,
        profile_settings,
    )
    visual_root.mkdir(parents=True, exist_ok=True)
    candidate_id = _next_candidate_id(visual_root)
    candidate_dir = visual_root / candidate_id
    candidate_dir.mkdir()
    staging_dir = candidate_dir / ".generation_staging"

    generator = generation_function or _default_generate
    settings = {
        "seed": seed,
        "steps": profile_settings["steps"],
        "strength": profile_settings["strength"],
        "guidance_scale": profile_settings["guidance_scale"],
        "control_scale": profile_settings["control_scale"],
    }
    if generation_settings:
        settings.update(generation_settings)
        settings["seed"] = seed

    try:
        generated = generator(
            Path(manual["confirmed_image"]),
            manual["analysis"],
            staging_dir,
            **settings,
        )
        if not isinstance(generated, dict):
            raise TypeError("generation_function must return a dictionary")

        portrait_path = _move_generated_file(
            generated["portrait"],
            candidate_dir / "portrait.png",
            "portrait",
        )
        control_path = _move_generated_file(
            generated["control_image"],
            candidate_dir / "control.png",
            "control image",
        )
        metadata_source = generated.get("metadata_file")
        private_metadata = _read_json(metadata_source)
        Path(metadata_source).unlink()

        color_policy = _enforce_color_policy(
            portrait_path,
            manual["analysis"],
        )
        private_metadata.update(
            {
                "portrait": str(portrait_path),
                "control_image": str(control_path),
                "seed": seed,
                "seed_visibility": "private_audit_metadata_only",
                "color_policy": color_policy,
                "generation_profile": {
                    "name": profile,
                    **profile_settings,
                },
                "terminal_experiment": profile == "ultra_locked",
            }
        )
        private_metadata_path = candidate_dir / "generation_private.json"
        _write_json(private_metadata_path, private_metadata)

        checks = build_review_checks(manual)
        checklist = {
            "confirmation_id": confirmation_id,
            "candidate_id": candidate_id,
            "instructions": (
                "Set every decision to true or false after comparing portrait.png "
                "with manual_confirmed_reference.png."
            ),
            "checks": [
                {**deepcopy(check), "decision": None, "comment": ""}
                for check in checks
            ],
        }
        checklist_path = candidate_dir / "review_checklist.json"
        _write_json(checklist_path, checklist)

        candidate = {
            "confirmation_id": confirmation_id,
            "candidate_id": candidate_id,
            "created_at": _now(),
            "status": STATUS_PENDING,
            "authoritative_reference": manual["confirmed_image"],
            "authoritative_reference_sha256": _sha256(
                manual["confirmed_image"]
            ),
            "portrait": str(portrait_path),
            "portrait_sha256": _sha256(portrait_path),
            "control_image": str(control_path),
            "private_generation_metadata": str(private_metadata_path),
            "review_checklist": str(checklist_path),
            "color_policy": color_policy,
            "generation_profile": {
                "name": profile,
                **profile_settings,
            },
            "terminal_experiment": profile == "ultra_locked",
            "required_checks": checks,
            "notice": NOTICE,
        }
        candidate_path = candidate_dir / "candidate.json"
        _write_json(candidate_path, candidate)
        candidate["candidate_file"] = str(candidate_path)
        return candidate
    except Exception:
        failure = {
            "confirmation_id": confirmation_id,
            "candidate_id": candidate_id,
            "failed_at": _now(),
            "status": "generation_failed",
            "notice": NOTICE,
        }
        _write_json(candidate_dir / "generation_failure.json", failure)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def load_visualization_candidate(
    confirmation_id,
    candidate_id,
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    candidate_id = _validate_candidate_id(candidate_id)
    path = (
        _visualization_root(confirmation_id, output_root)
        / candidate_id
        / "candidate.json"
    )
    candidate = _read_json(path)
    if candidate.get("confirmation_id") != confirmation_id:
        raise ValueError("Candidate confirmation_id does not match")
    if candidate.get("candidate_id") != candidate_id:
        raise ValueError("Candidate ID does not match")
    candidate["candidate_file"] = str(path)
    return candidate


def review_visualization_candidate(
    confirmation_id,
    candidate_id,
    reviewer,
    decisions,
    notes="",
    output_root=DEFAULT_CONFIRMATION_ROOT,
):
    """Accept only all-pass candidates; otherwise mark them rejected."""
    candidate = load_visualization_candidate(
        confirmation_id,
        candidate_id,
        output_root=output_root,
    )
    if candidate["status"] != STATUS_PENDING:
        raise ValueError("Only pending_review candidates can be reviewed")

    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2:
        raise ValueError("reviewer must contain at least 2 characters")
    if not isinstance(decisions, dict):
        raise TypeError("decisions must be a dictionary")
    if not isinstance(notes, str):
        raise TypeError("notes must be a string")

    required = {
        item["id"]: item
        for item in candidate["required_checks"]
    }
    if set(decisions) != set(required):
        missing = sorted(set(required) - set(decisions))
        extra = sorted(set(decisions) - set(required))
        raise ValueError(
            f"Review decisions must match required checks; missing={missing}, "
            f"extra={extra}"
        )
    if any(type(value) is not bool for value in decisions.values()):
        raise TypeError("Every review decision must be true or false")

    failed_ids = [
        check_id
        for check_id, passed in decisions.items()
        if not passed
    ]
    status = STATUS_REJECTED if failed_ids else STATUS_ACCEPTED
    review = {
        "confirmation_id": confirmation_id,
        "candidate_id": candidate_id,
        "reviewed_at": _now(),
        "reviewer": reviewer.strip(),
        "status": status,
        "decisions": decisions,
        "failed_checks": [required[check_id] for check_id in failed_ids],
        "notes": notes.strip(),
        "notice": NOTICE,
    }

    candidate_dir = Path(candidate["candidate_file"]).parent
    review_path = candidate_dir / "review.json"
    _write_json(review_path, review)

    candidate["status"] = status
    candidate["reviewed_at"] = review["reviewed_at"]
    candidate["review_file"] = str(review_path)
    candidate_path = Path(candidate.pop("candidate_file"))
    _write_json(candidate_path, candidate)

    review["review_file"] = str(review_path)
    return review
