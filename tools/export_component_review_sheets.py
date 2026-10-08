"""Export category-specific CSV files for reviewing component labels."""

import csv
import json
from pathlib import Path

from face_schema import FEATURE_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "static" / "assets" / "face_components" / "catalog.json"
OUTPUT_DIR = PROJECT_ROOT / "component_review_sheets"

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
    "glasses": ["glasses"],
    "marks": [],
}

BASE_FIELDS = [
    "id",
    "category",
    "file",
    "display_name",
]

ENDING_FIELDS = [
    "aliases",
    "review_status",
    "notes",
]


def allowed_values_text(field_name):
    values = FEATURE_SCHEMA[field_name]["allowed_values"]
    return " | ".join(values)


def export_category(category, entries):
    semantic_fields = CATEGORY_FIELDS[category]
    fieldnames = BASE_FIELDS + semantic_fields + ENDING_FIELDS
    output_path = OUTPUT_DIR / f"{category}_review.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for entry in entries:
            row = {
                "id": entry["id"],
                "category": category,
                "file": entry["file"],
                "display_name": entry["display_name"],
                "aliases": " | ".join(entry.get("aliases", [])),
                "review_status": entry.get("review_status", "pending"),
                "notes": "",
            }

            for field_name in semantic_fields:
                value = entry.get("attributes", {}).get(field_name, "")
                if isinstance(value, list):
                    value = " | ".join(value)
                row[field_name] = value

            writer.writerow(row)

    return output_path


def write_instructions():
    lines = [
        "THIRD EYE COMPONENT REVIEW INSTRUCTIONS",
        "=======================================",
        "",
        "1. Open one category CSV together with its contact-sheet image.",
        "2. Enter only values listed below. Do not invent new spellings.",
        "3. Use lowercase values exactly as shown.",
        "4. If uncertain, leave the value blank and keep review_status as pending.",
        "5. When reviewed, change review_status to reviewed.",
        "6. Separate multiple aliases with:  |  ",
        "7. Do not change id, category, or file.",
        "",
        "ALLOWED VALUES",
        "--------------",
    ]

    for category, fields in CATEGORY_FIELDS.items():
        if not fields:
            continue

        lines.append("")
        lines.append(category.upper())

        for field_name in fields:
            lines.append(
                f"- {field_name}: {allowed_values_text(field_name)}"
            )

    instructions_path = OUTPUT_DIR / "REVIEW_INSTRUCTIONS.txt"
    instructions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return instructions_path


def main():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalogue not found: {CATALOG_PATH}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    components = catalog["components"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = []

    for category, entries in components.items():
        if category not in CATEGORY_FIELDS:
            print(f"Skipping unknown category: {category}")
            continue

        output_path = export_category(category, entries)
        created.append(output_path)
        print(f"Created: {output_path}")

    instructions_path = write_instructions()
    print(f"Created: {instructions_path}")
    print()
    print(f"Created {len(created)} category review sheets.")


if __name__ == "__main__":
    main()
