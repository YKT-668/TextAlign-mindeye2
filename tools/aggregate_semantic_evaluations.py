#!/usr/bin/env python3
"""Aggregate non-identifying Cross-LLM, Human-written, or Human Audit results."""

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload["rows"]
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--group", default="condition")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[float]] = {}
    for path in args.inputs:
        for row in read_rows(path):
            grouped.setdefault(str(row[args.group]), []).append(float(row[args.metric]))

    output = [
        {
            args.group: group,
            "metric": args.metric,
            "mean": sum(values) / len(values),
            "n": len(values),
        }
        for group, values in sorted(grouped.items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
