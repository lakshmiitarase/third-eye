"""CLI for the read-only Third Eye Phase 13A catalogue audit."""

import argparse
from datetime import datetime
from pathlib import Path

from catalogue_geometry_validator import audit_catalogue, write_audit_reports

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = PROJECT_ROOT / "static" / "assets" / "face_components" / "catalog.json"
DEFAULT_REVIEWS = PROJECT_ROOT / "component_review_sheets"


def main():
    parser = argparse.ArgumentParser(
        description="Validate component assets, metadata, placements, and review-sheet consistency without modifying sources."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Path to catalog.json")
    parser.add_argument("--reviews", default=str(DEFAULT_REVIEWS), help="Path to completed component review CSVs")
    parser.add_argument("--output", help="New or empty audit output directory")
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "reports" / f"phase13a_catalogue_audit_{stamp}"

    report = audit_catalogue(args.catalog, review_root=args.reviews)
    paths = write_audit_reports(report, output_dir)
    summary = report["summary"]

    print("Phase 13A read-only catalogue audit complete.")
    print(f"Components: {summary['component_count']}")
    print(f"Referenced PNGs: {summary['referenced_png_count']}")
    print(f"Actual PNGs: {summary['actual_png_count']}")
    print(f"Review rows: {summary['review_row_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    print(f"Sources unchanged: {summary['sources_unchanged']}")
    print(f"JSON: {paths['json']}")
    print(f"CSV: {paths['csv']}")
    print(f"Summary: {paths['summary']}")

    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
