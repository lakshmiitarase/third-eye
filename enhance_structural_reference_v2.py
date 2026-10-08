"""Create a stronger deterministic V2 structural presentation copy."""

import argparse

from structural_enhancement_service import enhance_structural_reference_v2


def main():
    parser = argparse.ArgumentParser(
        description="Create a stronger non-generative structural presentation enhancement."
    )
    parser.add_argument("confirmation_id", help="Example: witness-78")
    args = parser.parse_args()

    record = enhance_structural_reference_v2(args.confirmation_id)
    print("V2 structural presentation enhancement complete.")
    print(f"Output: {record['output']}")
    print(f"Audit: {record['audit_file']}")
    print(f"Source SHA-256: {record['source_sha256']}")
    print(f"Output SHA-256: {record['output_sha256']}")
    print(f"Mean absolute grayscale difference: {record['mean_absolute_grayscale_difference']}")
    print("AI model used: no")
    print(record["notice"])


if __name__ == "__main__":
    main()
