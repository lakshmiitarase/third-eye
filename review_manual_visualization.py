"""Finalize a human checklist review for one visualization candidate."""

import argparse
import json
from pathlib import Path

from visualization_gate import review_visualization_candidate


def _load_checklist(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("Checklist must contain a checks list")

    decisions = {}
    comments = []
    for check in checks:
        check_id = check.get("id")
        decision = check.get("decision")
        if not isinstance(check_id, str):
            raise ValueError("Every checklist item requires an id")
        if type(decision) is not bool:
            raise ValueError(
                f"Checklist decision for {check_id} must be true or false"
            )
        decisions[check_id] = decision
        comment = str(check.get("comment", "")).strip()
        if comment:
            comments.append(f"{check_id}: {comment}")
    return decisions, "\n".join(comments)


def main():
    parser = argparse.ArgumentParser(
        description="Accept or reject a generated visualization after review."
    )
    parser.add_argument("confirmation_id", help="Example: witness-78")
    parser.add_argument("candidate_id", help="Example: candidate-001")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--checklist",
        required=True,
        help="Edited review_checklist.json with every decision set",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    decisions, checklist_comments = _load_checklist(args.checklist)
    notes = args.notes.strip()
    if checklist_comments:
        notes = "\n".join(part for part in (notes, checklist_comments) if part)

    review = review_visualization_candidate(
        args.confirmation_id,
        args.candidate_id,
        args.reviewer,
        decisions,
        notes=notes,
    )
    print(f"Review status: {review['status']}")
    if review["failed_checks"]:
        print("Candidate rejected because these checks failed:")
        for check in review["failed_checks"]:
            print(f"- {check['label']} (expected: {check['expected']})")
    print(review["notice"])


if __name__ == "__main__":
    main()
