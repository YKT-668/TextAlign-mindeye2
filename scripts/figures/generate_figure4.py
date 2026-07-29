#!/usr/bin/env python3
"""Generate the public Figure 4 comparison from an aggregate CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--value-column", default="mean")
    parser.add_argument("--output", type=Path, default=Path("results/figures/figure4.png"))
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.label_column not in frame or args.value_column not in frame:
        parser.error("input is missing the selected label or value column")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(frame[args.label_column].astype(str), frame[args.value_column].astype(float))
    axis.set_ylabel(args.value_column)
    axis.set_xlabel(args.label_column)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
