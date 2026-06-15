"""Run the Week 5 fuzzy scoring demo on the synthetic fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obpi.pipeline import run_scoring_pipeline  # noqa: E402


def main() -> int:
    """Score the synthetic metrics table and export membership metadata."""

    input_path = PROJECT_ROOT / "data" / "sample" / "synthetic_metrics.csv"
    output_path = PROJECT_ROOT / "results" / "week5_scored_metrics.csv"
    membership_report_path = (
        PROJECT_ROOT / "results" / "week5_membership_report.json"
    )

    summary = run_scoring_pipeline(
        input_path=input_path,
        output_path=output_path,
        membership_report_path=membership_report_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
