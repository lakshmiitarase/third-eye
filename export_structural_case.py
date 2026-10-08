"""Create a structural-only ZIP export for one fully reviewed case."""

import argparse

from case_export_service import export_structural_case


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export the authoritative structural composite and public audit "
            "trail without private AI generation data."
        )
    )
    parser.add_argument("confirmation_id", help="Example: witness-78")
    parser.add_argument("--exporter", required=True)
    args = parser.parse_args()

    result = export_structural_case(
        args.confirmation_id,
        exporter=args.exporter,
    )
    print("Structural-only case export complete.")
    print(f"ZIP: {result['zip_file']}")
    print(f"SHA-256: {result['zip_sha256']}")
    print(f"Policy status: {result['policy_status']}")
    print(f"AI candidates summarized: {result['ai_candidate_count']}")
    print("Private generation data included: no")
    print(result["notice"])


if __name__ == "__main__":
    main()
