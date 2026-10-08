"""Generate three photorealistic faces from exact-match structural variants."""

import argparse
from pathlib import Path

from lineup_service import load_lineup_context
from photorealistic_generator import (
    generate_photorealistic,
    unload_pipeline,
)


DEFAULT_SEEDS = (9201, 9202, 9203)


def generate_lineup_variants(confirmation_id, seeds=DEFAULT_SEEDS):
    lineup = load_lineup_context(confirmation_id)
    variants = lineup["variants"]

    if len(variants) != len(seeds):
        raise ValueError(
            f"Expected {len(seeds)} structural variants; found {len(variants)}"
        )

    lineup_directory = Path(lineup["context_file"]).parent
    output_root = lineup_directory / "photorealistic_lineup"
    results = []

    try:
        for variant, seed in zip(variants, seeds):
            variant_id = variant["variant_id"]
            reference_path = Path(variant["image"])
            output_directory = output_root / variant_id
            portrait_path = output_directory / f"portrait_seed_{seed}.png"

            if portrait_path.exists():
                print(f"Skipping existing {variant_id}: {portrait_path}")
                continue

            print(
                f"Generating {variant_id} from {reference_path} "
                f"with seed {seed}..."
            )

            record = generate_photorealistic(
                reference_path,
                lineup["analysis"],
                output_directory,
                seed=seed,
                steps=28,
                strength=0.65,
                guidance_scale=5.5,
                control_scale=0.75,
            )
            results.append(record)
            print(f"Completed {variant_id}: {record['portrait']}")
    finally:
        unload_pipeline()

    print("Photorealistic lineup generation finished.")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "confirmation_id",
        nargs="?",
        default="witness-74",
    )
    args = parser.parse_args()
    generate_lineup_variants(args.confirmation_id)


if __name__ == "__main__":
    main()
