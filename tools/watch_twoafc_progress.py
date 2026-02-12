#!/usr/bin/env python
# coding: utf-8
"""watch_twoafc_progress.py

Simple progress watcher for shared982 2AFC runs.

Usage:
  python tools/watch_twoafc_progress.py \
    --csv cache/model_eval_results/shared982_twoafc/twoafc_summary.csv \
    --log cache/model_eval_results/shared982_twoafc/twoafc_watcher.out

Then in another terminal:
  tail -f cache/model_eval_results/shared982_twoafc/twoafc_watcher.out
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def _read_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cache/model_eval_results/shared982_twoafc/twoafc_summary.csv")
    ap.add_argument("--log", default="cache/model_eval_results/shared982_twoafc/twoafc_watcher.out")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()

    while True:
        rows = _read_rows(csv_path)
        new = []
        for r in rows:
            key = r.get("metrics_json", "")
            if key and key not in seen:
                seen.add(key)
                new.append(r)

        if new:
            lines = []
            lines.append(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} new results ({len(new)}) ===")
            for r in new[-10:]:
                lines.append(
                    f"{r.get('group')} subj{r.get('subj')} {r.get('eval_repr')} tag={r.get('tag')} "
                    f"N={r.get('N')} B->I={float(r.get('twoafc_fwd_mean')):.3f} I->B={float(r.get('twoafc_bwd_mean')):.3f}"
                )
            lines.append("")
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
