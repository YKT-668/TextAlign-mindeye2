#!/usr/bin/env python3
"""Query the machine-readable public results catalog."""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="results/catalog/results.json")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    with open(args.catalog, encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        if args.experiment is None or row["experiment"] == args.experiment:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
