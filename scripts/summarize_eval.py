#!/usr/bin/env python3
"""Print benchmark scores from a training run's metrics.jsonl.

Avg@N is mean correctness across N sampled answers per problem. Pass@N is
the fraction of problems with at least one correct answer among those N answers.
Both are shown as percentages. The backend's bootstrap best@N is not Pass@N.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


AVERAGE_KEY = re.compile(r"^val-core/([^/]+)/acc/mean@(\d+)$")
BENCHMARK_ORDER = {name: index for index, name in enumerate(("AMC23", "AIME24", "AIME25"))}


def read_evaluations(path: Path) -> list[dict]:
    """Read evaluation rows, preserving evaluation order and any missing Pass@N."""
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metrics = record["data"]
                for key, value in metrics.items():
                    match = AVERAGE_KEY.fullmatch(key)
                    if match is None:
                        continue
                    source, sample_count = match.groups()
                    pass_value = metrics.get(f"val-core/{source}/acc/pass@{sample_count}")
                    average = float(value)
                    passed = None if pass_value is None else float(pass_value)
                    for score in (average, passed):
                        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                            raise ValueError("accuracy must be between 0 and 1")
                    if int(sample_count) < 1:
                        raise ValueError("sample count must be positive")
                    rows.append(
                        {
                            "step": record["step"],
                            "benchmark": source,
                            "samples": int(sample_count),
                            "average": average,
                            "pass": passed,
                        }
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid metric record: {exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Path to outputs/<run>/metrics.jsonl")
    parser.add_argument("--all", action="store_true", help="Show every evaluation, including step 0 (default: latest).")
    args = parser.parse_args(argv)
    try:
        rows = read_evaluations(args.metrics)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("No validation accuracy metrics found; wait for the first evaluation to finish.")
    if not args.all:
        rows = [row for row in rows if row["step"] == rows[-1]["step"]]
    rows.sort(key=lambda row: (row["step"], BENCHMARK_ORDER.get(row["benchmark"], 99), row["benchmark"], row["samples"]))
    print("| Step | Benchmark | N | Avg@N (%) | Pass@N (%) |")
    print("| ---: | :--- | ---: | ---: | ---: |")
    for row in rows:
        passed = "n/a" if row["pass"] is None else f"{100 * row['pass']:.2f}"
        print(f"| {row['step']} | {row['benchmark']} | {row['samples']} | {100 * row['average']:.2f} | {passed} |")


if __name__ == "__main__":
    main()
