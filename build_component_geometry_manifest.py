"""CLI for Third Eye Phase 13B provisional geometry outputs."""

import argparse
from pathlib import Path

from component_geometry_manifest import build_geometry_manifest, write_geometry_outputs

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = PROJECT_ROOT / "static" / "assets" / "face_components" / "catalog.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "static" / "assets" / "face_components" / "geometry_manifest.provisional.json"
DEFAULT_WORKSHEETS = PROJECT_ROOT / "component_geometry_review_sheets"


def main():
    parser = argparse.ArgumentParser(
        description="Create machine-measured geometry and blank expert landmark worksheets without modifying source assets."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--worksheets", default=str(DEFAULT_WORKSHEETS))
    args = parser.parse_args()

    manifest = build_geometry_manifest(args.catalog)
    paths = write_geometry_outputs(manifest, args.manifest, args.worksheets)
    summary = manifest["summary"]

    print("Phase 13B provisional geometry manifest created.")
    print(f"Components measured: {summary['component_count']}")
    print(f"Expert-reviewed components: {summary['expert_reviewed_component_count']}")
    print(f"Sources unchanged: {summary['sources_unchanged']}")
    print(f"Manifest: {paths['manifest']}")
    print(f"Geometry worksheets: {len(paths['worksheets'])}")
    print("Landmarks remain blank pending forensic-artist/domain-expert review.")
    print("Component selection behavior is unchanged in Phase 13B.")


if __name__ == "__main__":
    main()
