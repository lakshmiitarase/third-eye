"""Generate one local-only pending-review visualization.

The random diffusion seed is generated automatically and is retained only in
private generation metadata. It is never included in display filenames.
"""

import argparse

from visualization_gate import (
    GENERATION_PROFILES,
    generate_pending_visualization,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one optional AI visualization from the authoritative "
            "manual structural reference."
        )
    )
    parser.add_argument("confirmation_id", help="Example: witness-78")
    parser.add_argument(
        "--profile",
        choices=sorted(GENERATION_PROFILES),
        default="conservative",
        help="Generation profile; conservative is the default",
    )
    args = parser.parse_args()

    candidate = generate_pending_visualization(
        args.confirmation_id,
        profile=args.profile,
    )
    print("Generation complete; human review is required.")
    print(f"Candidate: {candidate['candidate_id']}")
    print(f"Portrait: {candidate['portrait']}")
    print(f"Checklist: {candidate['review_checklist']}")
    print(f"Profile: {candidate['generation_profile']['name']}")
    print(f"Status: {candidate['status']}")
    print(candidate["notice"])


if __name__ == "__main__":
    main()
