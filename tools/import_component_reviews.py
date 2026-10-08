"""Validate and import component review CSV files into catalog.json."""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from face_schema import FEATURE_SCHEMA


CATALOG_PATH = PROJECT_ROOT / "static" / "assets" / "face_components" / "catalog.json"
REVIEW_DIR = PROJECT_ROOT / "component_review_sheets"

CATEGORY_FIELDS = {
    "heads": ["face_shape", "face_width", "jaw_shape", "chin_shape"],
    "hair": ["hair_length", "hair_texture", "hairline"],
    "eyes": ["eye_shape", "eye_size", "eye_spacing"],
    "eyebrows": ["eyebrow_shape", "eyebrow_thickness", "eyebrow_spacing"],
    "noses": ["nose_width", "nose_length", "nose_bridge", "nose_tip"],
    "lips": ["lip_thickness", "mouth_width"],
    "moustaches": ["moustache_style"],
    "beards": ["beard_style"],
    "ears": ["ear_size"],
    "necks": ["neck_width"],
}

ALLOWED_REVIEW_STATUSES = {"provisional", "reviewed"}


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def split_aliases(value):
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def validate_and_prepare(catalog):
    errors = []
    updates = {}
    total_rows = 0

    for category, semantic_fields in CATEGORY_FIELDS.items():
        csv_path = REVIEW_DIR / f"{category}_review.csv"

        if not csv_path.exists():
            errors.append(f"Missing review file: {csv_path}")
            continue

        rows = read_csv_rows(csv_path)
        catalog_entries = catalog["components"].get(category, [])
        catalog_by_id = {entry["id"]: entry for entry in catalog_entries}
        csv_ids = {row.get("id", "").strip() for row in rows}
        catalog_ids = set(catalog_by_id.keys())

        if csv_ids != catalog_ids:
            missing = sorted(catalog_ids - csv_ids)
            unexpected = sorted(csv_ids - catalog_ids)
            if missing:
                errors.append(f"{category}: missing CSV IDs: {missing}")
            if unexpected:
                errors.append(f"{category}: unexpected CSV IDs: {unexpected}")

        category_updates = {}

        for row_number, row in enumerate(rows, start=2):
            component_id = row.get("id", "").strip()
            row_category = row.get("category", "").strip()
            row_file = row.get("file", "").strip()
            review_status = row.get("review_status", "").strip()

            if not component_id:
                errors.append(f"{csv_path.name}:{row_number}: missing id")
                continue

            if component_id not in catalog_by_id:
                continue

            catalog_entry = catalog_by_id[component_id]

            if row_category != category:
                errors.append(
                    f"{csv_path.name}:{row_number}: category must be '{category}', "
                    f"found '{row_category}'"
                )

            if row_file != catalog_entry["file"]:
                errors.append(
                    f"{csv_path.name}:{row_number}: file must be "
                    f"'{catalog_entry['file']}', found '{row_file}'"
                )

            if review_status not in ALLOWED_REVIEW_STATUSES:
                errors.append(
                    f"{csv_path.name}:{row_number}: invalid review_status "
                    f"'{review_status}'"
                )

            attributes = {}

            for field_name in semantic_fields:
                value = row.get(field_name, "").strip()

                if not value:
                    errors.append(
                        f"{csv_path.name}:{row_number}: missing value for {field_name}"
                    )
                    continue

                allowed_values = FEATURE_SCHEMA[field_name]["allowed_values"]

                if value not in allowed_values:
                    errors.append(
                        f"{csv_path.name}:{row_number}: invalid {field_name} "
                        f"value '{value}'. Allowed: {allowed_values}"
                    )
                    continue

                attributes[field_name] = value

            category_updates[component_id] = {
                "display_name": row.get("display_name", "").strip()
                or catalog_entry["display_name"],
                "attributes": attributes,
                "aliases": split_aliases(row.get("aliases", "")),
                "review_status": review_status,
                "review_notes": row.get("notes", "").strip(),
            }

        updates[category] = category_updates
        total_rows += len(rows)
        print(f"Validated {category}: {len(rows)} rows")

    return errors, updates, total_rows


def apply_updates(catalog, updates):
    updated_count = 0

    for category, category_updates in updates.items():
        for entry in catalog["components"][category]:
            component_id = entry["id"]
            if component_id not in category_updates:
                continue

            update = category_updates[component_id]
            entry["display_name"] = update["display_name"]
            entry["attributes"] = update["attributes"]
            entry["aliases"] = update["aliases"]
            entry["review_status"] = update["review_status"]
            entry["review_notes"] = update["review_notes"]
            updated_count += 1

    catalog["catalog_status"] = "provisional_semantic_labels"
    catalog["semantic_review"] = {
        "status": "provisional",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "Visual labels are provisional until confirmed by a domain reviewer.",
    }

    return updated_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write validated review data into catalog.json",
    )
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalogue not found: {CATALOG_PATH}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    errors, updates, total_rows = validate_and_prepare(catalog)

    print()
    print(f"Total reviewed rows found: {total_rows}")

    if errors:
        print()
        print(f"VALIDATION FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed.")

    if not args.apply:
        print("Dry run only: catalog.json was not changed.")
        print("Run again with --apply to update the catalogue.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = CATALOG_PATH.with_name(f"catalog.backup-{timestamp}.json")
    shutil.copy2(CATALOG_PATH, backup_path)

    updated_count = apply_updates(catalog, updates)

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Backup created: {backup_path}")
    print(f"Updated components: {updated_count}")
    print(f"Catalogue written: {CATALOG_PATH}")

    if updated_count != 88:
        raise RuntimeError(f"Expected 88 updates but applied {updated_count}.")


if __name__ == "__main__":
    main()
