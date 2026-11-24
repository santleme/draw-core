"""Check identity-residual shifts against independent static ink masks.

This applies known integer translations to human-reviewed holdout masks. It
tests the estimator's geometric response on real handwriting shapes, but it
does not simulate camera noise, a moving sheet, tracker loss/reacquisition,
or anchor arbitration in a temporal capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from draw.ink.worker import _estimate_identity_residual_evidence

_SHIFTS = (
    (6, 0),
    (-6, 0),
    (0, 6),
    (0, -6),
    (6, 6),
    (-6, 6),
    (6, -6),
    (-6, -6),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_holdout(root: Path) -> list[tuple[str, str, np.ndarray]]:
    manifest_path = root / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = []
    for row in rows:
        source = row.get("source") or {}
        if (
            row.get("split") != "holdout"
            or row.get("origin") != "independent_label"
            or row.get("label_source") != "human_from_scratch"
            or row.get("label_reviewed") is not True
            or source.get("prior_training") is not False
        ):
            raise ValueError(
                f"not an independent reviewed holdout row: {row.get('file')}"
            )

        filename = row["file"]
        image_path = root / "input" / filename
        mask_path = root / "mask" / filename
        if _sha256(image_path) != row.get("input_sha256"):
            raise ValueError(f"input SHA-256 mismatch: {filename}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"missing image or mismatched mask dimensions: {filename}")
        if not np.isin(mask, (0, 255)).all():
            raise ValueError(f"mask is not binary: {filename}")
        cases.append((row["session_id"], filename, mask))

    if not cases:
        raise ValueError(f"no holdout pages in {manifest_path}")
    return cases


def run(root: Path, hidden_strip_px: int) -> dict[str, object]:
    cases = _load_holdout(root)
    per_regime: dict[str, Counter[str]] = defaultdict(Counter)
    per_session: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    per_shift: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)

    for session, _filename, mask in cases:
        reference = mask.copy()
        height, width = reference.shape
        if not 0 <= hidden_strip_px < width:
            raise ValueError(
                f"hidden strip must be in [0, {width}), got {hidden_strip_px}"
            )

        for dx, dy in _SHIFTS:
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            evidence = cv2.warpAffine(
                reference,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)
            for regime in ("full", "left_strip_hidden"):
                coverage = np.ones(reference.shape, dtype=bool)
                if regime == "left_strip_hidden":
                    coverage[:, :hidden_strip_px] = False
                result = _estimate_identity_residual_evidence(
                    reference,
                    None,
                    evidence,
                    None,
                    None,
                    0.5,
                    coverage,
                )
                expected = (-dx, -dy)
                outcome = (
                    "correct"
                    if result.shift == expected
                    else "abstain"
                    if result.shift == (0, 0)
                    else "wrong"
                )
                per_regime[regime][outcome] += 1
                per_session[(regime, session)][outcome] += 1
                per_shift[(regime, dx, dy)][outcome] += 1

    return {
        "holdout_root": str(root),
        "pages": len(cases),
        "sessions": sorted({session for session, _, _ in cases}),
        "shifts_px": [list(shift) for shift in _SHIFTS],
        "coverage": {
            "full": "all pixels observed",
            "left_strip_hidden": f"x < {hidden_strip_px} hidden; remaining pixels observed",
        },
        "transform": "cv2.warpAffine, INTER_NEAREST, zero border; known integer translation",
        "trials": len(cases) * len(_SHIFTS) * 2,
        "outcomes_by_regime": {
            regime: dict(sorted(counts.items()))
            for regime, counts in sorted(per_regime.items())
        },
        "outcomes_by_session": {
            f"{regime}/{session}": dict(sorted(counts.items()))
            for (regime, session), counts in sorted(per_session.items())
        },
        "outcomes_by_shift": {
            f"{regime}/{dx},{dy}": dict(sorted(counts.items()))
            for (regime, dx, dy), counts in sorted(per_shift.items())
        },
        "limitations": [
            "The reviewed labels are static masks; transforms are simulated.",
            "No camera noise, perspective change, hand motion, tracker loss/reacquisition, or temporal arbitration is evaluated.",
            "An abstention is reported separately from an incorrect nonzero estimate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-root",
        type=Path,
        default=Path("~/draw-data/holdout-v1").expanduser(),
    )
    parser.add_argument("--hidden-strip-px", type=int, default=140)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.holdout_root, args.hidden_strip_px), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
