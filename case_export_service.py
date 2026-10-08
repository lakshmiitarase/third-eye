"""Structural-only case export and audit reporting.

This service never imports or loads AI generation libraries. It exports the
manually confirmed structural reference, public review outcomes, and integrity
checks while excluding private seeds, prompts, model metadata, control images,
and rejected AI portraits.
"""

import hashlib
import html
import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from confirmation_service import DEFAULT_CONFIRMATION_ROOT
from face_composer import validate_case_id


EXPORT_SCHEMA_VERSION = "1.0"
EXPORT_MODE = "structural_only_case_export"
CASE_POSITIONING = (
    "A witness-assisted forensic facial composite system that generates "
    "plausible candidates—not an exact identification system."
)
AUTHORITATIVE_NOTICE = (
    "The manually confirmed structural composite is the only authoritative "
    "witness-assisted output in this package."
)
AI_EXCLUSION_NOTICE = (
    "Rejected AI portraits, private diffusion seeds, prompts, model metadata, "
    "and control images are intentionally excluded from this export."
)
LABEL_NOTICE = (
    "Semantic component labels are provisional until reviewed by a forensic "
    "artist or other qualified domain expert."
)


def _now():
    return datetime.now().astimezone()


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_directory(confirmation_id, output_root):
    confirmation_id = validate_case_id(confirmation_id)
    return Path(output_root) / confirmation_id / "manual_editor"


def _load_manual_record(confirmation_id, output_root):
    manual_dir = _manual_directory(confirmation_id, output_root)
    record_path = manual_dir / "manual_selection.json"
    record = _read_json(record_path)
    if record.get("confirmation_id") != confirmation_id:
        raise ValueError("Manual selection confirmation_id does not match")
    if record.get("mode") != "witness_assisted_manual_component_editor":
        raise ValueError("Unsupported manual selection mode")

    image_path = manual_dir / "manual_confirmed_reference.png"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Authoritative structural reference not found: {image_path}"
        )
    return manual_dir, record, image_path


def _public_features(analysis):
    features = analysis.get("features", {}) if isinstance(analysis, dict) else {}
    result = []
    for name, state in sorted(features.items()):
        if not isinstance(state, dict):
            continue
        value = state.get("value")
        if value in (None, "", "unknown"):
            continue
        result.append(
            {
                "name": name,
                "value": value,
                "negated": bool(state.get("negated")),
                "source_text": state.get("source_text", ""),
            }
        )
    return result


def _load_candidate_summaries(manual_dir):
    visualizations = Path(manual_dir) / "visualizations"
    if not visualizations.is_dir():
        return []

    summaries = []
    for candidate_dir in sorted(visualizations.iterdir()):
        if not candidate_dir.is_dir():
            continue
        candidate_path = candidate_dir / "candidate.json"
        if not candidate_path.is_file():
            continue

        candidate = _read_json(candidate_path)
        status = candidate.get("status")
        if status == "pending_review":
            raise ValueError(
                f"Candidate {candidate_dir.name} is still pending review; "
                "complete its review before exporting the case"
            )
        if status not in {"accepted", "rejected"}:
            raise ValueError(
                f"Candidate {candidate_dir.name} has unsupported status: "
                f"{status}"
            )

        review_path = candidate_dir / "review.json"
        review = _read_json(review_path)
        if review.get("status") != status:
            raise ValueError(
                f"Candidate and review status differ for {candidate_dir.name}"
            )

        failed_checks = []
        for check in review.get("failed_checks", []):
            failed_checks.append(
                {
                    "id": check.get("id"),
                    "label": check.get("label"),
                    "expected": check.get("expected"),
                }
            )

        profile = candidate.get("generation_profile") or {}
        summaries.append(
            {
                "candidate_id": candidate.get(
                    "candidate_id",
                    candidate_dir.name,
                ),
                "status": status,
                "created_at": candidate.get("created_at"),
                "reviewed_at": review.get("reviewed_at"),
                "reviewer": review.get("reviewer"),
                "profile": profile.get("name", "legacy_unspecified"),
                "terminal_experiment": bool(
                    candidate.get("terminal_experiment")
                ),
                "failed_checks": failed_checks,
                "notes": review.get("notes", ""),
                "portrait_included": False,
                "private_generation_metadata_included": False,
            }
        )
    return summaries


def _case_outcome(candidate_summaries):
    accepted = [
        item for item in candidate_summaries if item["status"] == "accepted"
    ]
    rejected = [
        item for item in candidate_summaries if item["status"] == "rejected"
    ]
    if accepted:
        return {
            "policy_status": "structural_authoritative_with_optional_accepted_visualization",
            "accepted_ai_candidates": len(accepted),
            "rejected_ai_candidates": len(rejected),
        }
    return {
        "policy_status": "structural_only_no_ai_candidate_accepted",
        "accepted_ai_candidates": 0,
        "rejected_ai_candidates": len(rejected),
    }


def _escape(value):
    return html.escape("" if value is None else str(value))


def _render_report(case_record):
    features = "\n".join(
        "<tr>"
        f"<td>{_escape(item['name'].replace('_', ' ').title())}</td>"
        f"<td>{_escape(str(item['value']).replace('_', ' '))}</td>"
        f"<td>{'Yes' if item['negated'] else 'No'}</td>"
        "</tr>"
        for item in case_record["controlled_attributes"]
    ) or '<tr><td colspan="3">No controlled attributes recorded.</td></tr>'

    choices = "\n".join(
        "<tr>"
        f"<td>{_escape(category.replace('_', ' ').title())}</td>"
        f"<td>{_escape(component_id)}</td>"
        "</tr>"
        for category, component_id in sorted(
            case_record["manual_selection"]["choices"].items()
        )
    ) or '<tr><td colspan="2">No component choices recorded.</td></tr>'

    candidate_rows = []
    for candidate in case_record["ai_candidate_reviews"]:
        failed = "; ".join(
            item.get("label") or item.get("id") or "Unnamed check"
            for item in candidate["failed_checks"]
        ) or "None"
        candidate_rows.append(
            "<tr>"
            f"<td>{_escape(candidate['candidate_id'])}</td>"
            f"<td class=\"status-{_escape(candidate['status'])}\">"
            f"{_escape(candidate['status'].upper())}</td>"
            f"<td>{_escape(candidate['profile'])}</td>"
            f"<td>{_escape(candidate['reviewer'])}</td>"
            f"<td>{_escape(failed)}</td>"
            "</tr>"
        )
    candidates = "\n".join(candidate_rows) or (
        '<tr><td colspan="5">No AI candidates were generated.</td></tr>'
    )

    contradictions = case_record["manual_selection"]["contradictions"]
    if contradictions:
        contradiction_html = "<ul>" + "".join(
            "<li>"
            f"{_escape(item.get('category_label', item.get('category')))}: "
            f"{_escape(item.get('display_name', item.get('component_id')))}"
            "</li>"
            for item in contradictions
        ) + "</ul>"
    else:
        contradiction_html = "<p>None recorded.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structural Case Report — {_escape(case_record['confirmation_id'])}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #eef2f7; color: #172033; }}
main {{ max-width: 980px; margin: 28px auto; background: white; padding: 34px; box-shadow: 0 8px 30px #0002; }}
h1, h2 {{ color: #12345b; }}
.notice {{ border-left: 5px solid #246fba; padding: 12px 16px; background: #edf6ff; margin: 14px 0; }}
.warning {{ border-left-color: #b56a00; background: #fff5df; }}
.authoritative {{ display: block; max-width: 520px; width: 100%; margin: 18px auto; border: 1px solid #ccd5e0; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ccd5e0; padding: 9px; text-align: left; vertical-align: top; }}
th {{ background: #e8eef6; }}
.status-rejected {{ color: #a3212b; font-weight: bold; }}
.status-accepted {{ color: #167044; font-weight: bold; }}
.meta {{ color: #5b6574; font-size: .92rem; }}
@media print {{ body {{ background: white; }} main {{ box-shadow: none; margin: 0; max-width: none; }} }}
</style>
</head>
<body><main>
<h1>Structural Case Report</h1>
<p class="meta">Case: <strong>{_escape(case_record['confirmation_id'])}</strong><br>
Exported: {_escape(case_record['exported_at'])}<br>
Exporter: {_escape(case_record['exporter'])}</p>
<div class="notice"><strong>System positioning:</strong> {_escape(CASE_POSITIONING)}</div>
<div class="notice"><strong>Authoritative output:</strong> {_escape(AUTHORITATIVE_NOTICE)}</div>
<div class="notice warning"><strong>AI exclusion:</strong> {_escape(AI_EXCLUSION_NOTICE)}</div>
<h2>Witness description</h2>
<p>{_escape(case_record['description'])}</p>
<h2>Authoritative structural reference</h2>
<img class="authoritative" src="authoritative_structural_reference.png" alt="Authoritative manually confirmed structural reference">
<p>SHA-256: <code>{_escape(case_record['authoritative_reference']['sha256'])}</code></p>
<h2>Controlled attributes</h2>
<table><thead><tr><th>Attribute</th><th>Value</th><th>Negated</th></tr></thead><tbody>{features}</tbody></table>
<h2>Manual component selections</h2>
<table><thead><tr><th>Category</th><th>Component ID</th></tr></thead><tbody>{choices}</tbody></table>
<h2>Manual contradictions</h2>
{contradiction_html}
<p>Reason: {_escape(case_record['manual_selection']['contradiction_reason'] or 'Not applicable')}</p>
<h2>AI candidate review outcomes</h2>
<table><thead><tr><th>Candidate</th><th>Status</th><th>Profile</th><th>Reviewer</th><th>Failed checks</th></tr></thead><tbody>{candidates}</tbody></table>
<p><strong>Policy outcome:</strong> {_escape(case_record['outcome']['policy_status'])}</p>
<div class="notice warning">{_escape(LABEL_NOTICE)}</div>
<p class="meta">Rejected AI portraits and all private generation data are excluded. Refer to case_audit.json for the structured public audit summary and checksums.sha256 for integrity verification.</p>
</main></body></html>"""


def _write_checksums(build_dir):
    build_dir = Path(build_dir)
    lines = []
    for path in sorted(item for item in build_dir.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(build_dir).as_posix()}")
    checksum_path = build_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _create_zip(build_dir, zip_path, package_name):
    build_dir = Path(build_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(item for item in build_dir.rglob("*") if item.is_file()):
            archive.write(
                path,
                arcname=(Path(package_name) / path.relative_to(build_dir)).as_posix(),
            )
    return zip_path


def export_structural_case(
    confirmation_id,
    exporter,
    output_root=DEFAULT_CONFIRMATION_ROOT,
    export_root=None,
):
    """Create a seed-free structural case ZIP and an internal export audit."""
    confirmation_id = validate_case_id(confirmation_id)
    if not isinstance(exporter, str) or len(exporter.strip()) < 2:
        raise ValueError("exporter must contain at least 2 characters")

    manual_dir, manual, authoritative_image = _load_manual_record(
        confirmation_id,
        output_root,
    )
    candidates = _load_candidate_summaries(manual_dir)
    exported_at = _now()
    timestamp = exported_at.strftime("%Y%m%d-%H%M%S")
    export_id = f"structural-case-export-{timestamp}"
    package_name = f"{confirmation_id}_{export_id}"

    if export_root is None:
        project_root = Path(output_root).parent.parent.parent
        export_root = project_root / "exports"
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    zip_path = export_root / f"{package_name}.zip"
    if zip_path.exists():
        raise FileExistsError(f"Export already exists: {zip_path}")

    case_record = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_mode": EXPORT_MODE,
        "export_id": export_id,
        "confirmation_id": confirmation_id,
        "exported_at": exported_at.isoformat(timespec="seconds"),
        "exporter": exporter.strip(),
        "positioning": CASE_POSITIONING,
        "description": manual.get("description", ""),
        "authoritative_reference": {
            "file": "authoritative_structural_reference.png",
            "sha256": _sha256(authoritative_image),
            "status": "authoritative_witness_assisted_structural_composite",
        },
        "controlled_attributes": _public_features(
            manual.get("analysis", {})
        ),
        "manual_selection": {
            "finalized_at": manual.get("finalized_at"),
            "choices": manual.get("choices", {}),
            "contradictions": manual.get("contradictions", []),
            "contradiction_reason": manual.get("contradiction_reason"),
        },
        "ai_candidate_reviews": candidates,
        "outcome": _case_outcome(candidates),
        "exclusions": {
            "rejected_ai_portraits": True,
            "private_diffusion_seeds": True,
            "prompts": True,
            "model_metadata": True,
            "control_images": True,
        },
        "notices": [
            AUTHORITATIVE_NOTICE,
            AI_EXCLUSION_NOTICE,
            LABEL_NOTICE,
        ],
    }

    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{package_name}-", dir=export_root)
    )
    try:
        shutil.copy2(
            authoritative_image,
            build_dir / "authoritative_structural_reference.png",
        )
        _write_json(build_dir / "case_audit.json", case_record)
        (build_dir / "case_report.html").write_text(
            _render_report(case_record),
            encoding="utf-8",
        )
        (build_dir / "README.txt").write_text(
            "\n".join(
                [
                    "THIRD EYE — STRUCTURAL-ONLY CASE EXPORT",
                    "",
                    CASE_POSITIONING,
                    "",
                    AUTHORITATIVE_NOTICE,
                    AI_EXCLUSION_NOTICE,
                    LABEL_NOTICE,
                    "",
                    "Open case_report.html to view the report.",
                    "Use checksums.sha256 to verify package files.",
                ]
            ) + "\n",
            encoding="utf-8",
        )
        _write_checksums(build_dir)
        _create_zip(build_dir, zip_path, package_name)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    export_audit = {
        "export_id": export_id,
        "confirmation_id": confirmation_id,
        "exported_at": case_record["exported_at"],
        "exporter": exporter.strip(),
        "mode": EXPORT_MODE,
        "zip_file": str(zip_path),
        "zip_sha256": _sha256(zip_path),
        "policy_status": case_record["outcome"]["policy_status"],
        "ai_candidate_count": len(candidates),
        "private_generation_data_included": False,
        "notice": AUTHORITATIVE_NOTICE,
    }
    audit_path = manual_dir / "exports" / f"{export_id}.json"
    _write_json(audit_path, export_audit)

    export_audit["audit_file"] = str(audit_path)
    return export_audit
