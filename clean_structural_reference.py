"""Create one deterministic cleaned presentation copy of a manual reference."""

import argparse

from structural_cleanup_service import clean_structural_reference


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean the presentation of an authoritative structural reference "
            "without changing its geometry or using an AI model."
        )
    )
    parser.add_argument("confirmation_id", help="Example: witness-78")
    args = parser.parse_args()

    record = clean_structural_reference(args.confirmation_id)
    print("Structural presentation cleanup complete.")
    print(f"Output: {record['output']}")
    print(f"Audit: {record['audit_file']}")
    print(f"Source SHA-256: {record['source_sha256']}")
    print(f"Output SHA-256: {record['output_sha256']}")
    print(
        "Mean absolute grayscale difference: "
        f"{record['mean_absolute_grayscale_difference']}"
    )
    print("AI model used: no")
    print(record["notice"])


if __name__ == "__main__":
    main()
