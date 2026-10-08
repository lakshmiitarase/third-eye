"""Generate three lineup portraits with automatic private random seeds."""

import json
import secrets
from pathlib import Path

from lineup_service import load_lineup_context
from photorealistic_generator import (
    generate_photorealistic,
    unload_pipeline,
)


def _clean_output_names(record, output_directory):
    portrait_source = Path(record["portrait"])
    control_source = Path(record["control_image"])
    metadata_source = Path(record["metadata_file"])

    portrait_target = output_directory / "portrait.png"
    control_target = output_directory / "control.png"
    metadata_target = output_directory / "generation.json"

    portrait_source.replace(portrait_target)
    control_source.replace(control_target)

    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    metadata["portrait"] = str(portrait_target)
    metadata["control_image"] = str(control_target)
    metadata["seed_visibility"] = "private_audit_metadata_only"
    metadata["display_name"] = output_directory.name

    metadata_target.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_source.unlink()

    return {
        "portrait": str(portrait_target),
        "control_image": str(control_target),
        "metadata_file": str(metadata_target),
    }


def generate_lineup_faces(confirmation_id):
    lineup = load_lineup_context(confirmation_id)
    variants = lineup["variants"]

    if len(variants) != 3:
        raise ValueError(
            f"Expected 3 structural variants; found {len(variants)}"
        )

    lineup_directory = Path(lineup["context_file"]).parent
    output_root = lineup_directory / "photorealistic_lineup"
    completed = []

    try:
        for variant in variants:
            variant_id = variant["variant_id"]
            output_directory = output_root / variant_id
            final_portrait = output_directory / "portrait.png"

            if final_portrait.exists():
                print(f"Skipping existing {variant_id}: {final_portrait}")
                completed.append(str(final_portrait))
                continue

            # The seed is necessary for diffusion but is automatically chosen
            # and retained only in the private generation metadata.
            seed = secrets.randbelow(2**32)

            print(f"Generating {variant_id}...")
            record = generate_photorealistic(
                Path(variant["image"]),
                lineup["analysis"],
                output_directory,
                seed=seed,
                steps=28,
                strength=0.65,
                guidance_scale=5.5,
                control_scale=0.75,
            )

            clean = _clean_output_names(record, output_directory)
            completed.append(clean["portrait"])
            print(f"Completed {variant_id}: {clean['portrait']}")
    finally:
        unload_pipeline()

    print("All lineup portraits are complete.")
    return completed


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_id")
    args = parser.parse_args()
    generate_lineup_faces(args.confirmation_id)


if __name__ == "__main__":
    main()
