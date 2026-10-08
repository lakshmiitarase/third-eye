"""Initialize catalog.json for Third Eye facial components."""

import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ROOT = PROJECT_ROOT / "static" / "assets" / "face_components"
CATALOG_PATH = COMPONENT_ROOT / "catalog.json"

CANVAS_WIDTH = 620
CANVAS_HEIGHT = 620

# Original JavaFX positions on the 620 x 620 sketch canvas.
POSITIONS = {
    "heads": [
        (166, 82), (180, 115), (175, 136), (177, 127), (175, 113),
        (177, 132), (174, 127), (176, 112), (175, 109), (175, 109),
    ],
    "hair": [
        (57, 1), (82, 11), (30, 11), (123, 21), (81, 11), (150, 62),
        (162, 67), (118, 26), (101, 8), (157, 77), (155, 76), (81, 7),
    ],
    "eyes": [
        (208, 273), (215, 276), (211, 271), (215, 276), (211, 271),
        (215, 276), (211, 271), (215, 276), (211, 271), (215, 276),
        (211, 271), (214, 272),
    ],
    "eyebrows": [
        (207, 260), (194, 213), (204, 254), (210, 254), (202, 258),
        (203, 257), (201, 259), (200, 255), (203, 258), (203, 250),
        (208, 258), (213, 249),
    ],
    "noses": [
        (284, 323), (287, 327), (286, 318), (282, 327), (287, 326),
        (279, 319), (279, 319), (274, 319), (285, 312), (278, 315),
        (277, 318), (280, 322),
    ],
    "lips": [
        (270, 433), (269, 428), (274, 429), (262, 432), (263, 427),
        (264, 428), (278, 432), (277, 432), (244, 423), (245, 428),
        (261, 429), (264, 428),
    ],
    "moustaches": [
        (239, 397), (240, 398), (251, 400), (261, 400), (249, 399),
        (262, 401),
    ],
    "beards": [
        (247, 478), (196, 422), (185, 345), (180, 368), (187, 338),
        (180, 342),
    ],
    "ears": [
        (162, 298), (432, 301), (150, 291), (434, 298),
    ],
    "necks": [
        (202, 411), (213, 434),
    ],
    "glasses": [],
    "marks": [],
}

CATEGORY_PREFIXES = {
    "heads": "head",
    "hair": "hair",
    "eyes": "eyes",
    "eyebrows": "eyebrows",
    "noses": "nose",
    "lips": "lips",
    "moustaches": "moustache",
    "beards": "beard",
    "ears": "ear",
    "necks": "neck",
    "glasses": "glasses",
    "marks": "mark",
}

SOURCE_CATEGORY = {
    "heads": "head",
    "hair": "hair",
    "eyes": "eyes",
    "eyebrows": "eyebrows",
    "noses": "nose",
    "lips": "lips",
    "moustaches": "mustach",
    "beards": "mustach",
    "ears": "more",
    "necks": "more",
    "glasses": None,
    "marks": None,
}


def source_number(category, target_number):
    if category == "beards":
        return target_number + 6
    if category == "necks":
        return target_number + 4
    return target_number


def build_entry(category, path, index):
    with Image.open(path) as image:
        width, height = image.size
        image_mode = image.mode

    positions = POSITIONS[category]
    x, y = positions[index] if index < len(positions) else (None, None)
    target_number = index + 1
    original_number = source_number(category, target_number)
    source_category = SOURCE_CATEGORY[category]

    return {
        "id": path.stem,
        "display_name": path.stem.replace("_", " ").title(),
        "file": path.relative_to(COMPONENT_ROOT).as_posix(),
        "category": category,
        "source": {
            "project": "Third-Eye Java forensic sketch repository",
            "category": source_category,
            "original_file": (
                f"{original_number:02d}.png" if source_category else None
            ),
        },
        "image": {
            "width": width,
            "height": height,
            "mode": image_mode,
        },
        "placement": {
            "canvas_width": CANVAS_WIDTH,
            "canvas_height": CANVAS_HEIGHT,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "layer": category,
        },
        "attributes": {},
        "aliases": [],
        "review_status": "pending",
        "enabled": True,
    }


def main():
    if not COMPONENT_ROOT.exists():
        raise FileNotFoundError(f"Missing component root: {COMPONENT_ROOT}")

    components = {}
    total = 0

    for category, prefix in CATEGORY_PREFIXES.items():
        category_dir = COMPONENT_ROOT / category
        files = sorted(category_dir.glob(f"{prefix}_*.png"))
        entries = [
            build_entry(category, path, index)
            for index, path in enumerate(files)
        ]
        components[category] = entries
        total += len(entries)
        print(f"{category}: {len(entries)}")

    catalog = {
        "schema_version": "1.0",
        "catalog_status": "needs_semantic_review",
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "background": "white",
        },
        "layer_order": [
            "necks",
            "ears",
            "heads",
            "eyes",
            "eyebrows",
            "noses",
            "lips",
            "moustaches",
            "beards",
            "hair",
            "glasses",
            "marks",
        ],
        "components": components,
    }

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Wrote: {CATALOG_PATH}")
    print(f"Total catalogued assets: {total}")

    if total != 88:
        raise RuntimeError(f"Expected 88 assets but catalogued {total}.")


if __name__ == "__main__":
    main()
