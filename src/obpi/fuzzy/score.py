"""Command-line entrypoint for fuzzy OBPI scoring."""

from __future__ import annotations

import argparse

from obpi.fuzzy.scoring import score_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Score OBPI metrics with the fuzzy engine.")
    parser.add_argument("input_csv", help="CSV containing M1..M9 metric columns")
    parser.add_argument("output_csv", help="Path where scored CSV should be written")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Use default normalized memberships instead of calibrating from the input CSV",
    )
    args = parser.parse_args()

    scored = score_csv(
        args.input_csv,
        args.output_csv,
        calibrate_memberships=not args.no_calibration,
    )
    print(f"Wrote {len(scored)} scored rows to {args.output_csv}")


if __name__ == "__main__":
    main()
