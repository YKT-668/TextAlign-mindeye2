#!/usr/bin/env python
# coding: utf-8
"""Simple progress watcher for shared982 CCD runs.

Usage:
  python tools/rerun_all_ccd_shared982.py
  python tools/watch_ccd_progress.py \
    --csv cache/model_eval_results/shared982_ccd/ccd_summary.csv \
    --log cache/model_eval_results/shared982_ccd/ccd_watcher.out

Then in another shell:
  tail -f cache/model_eval_results/shared982_ccd/ccd_watcher.out
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


PROJ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(PROJ / "cache" / "model_eval_results" / "shared982_ccd" / "ccd_summary.csv"),
        help="Path to ccd_summary.csv",
    )
    ap.add_argument(
        "--log",
        default=str(PROJ / "cache" / "model_eval_results" / "shared982_ccd" / "ccd_watcher.out"),
        help="Output log file to append updates",
    )
    ap.add_argument("--poll", type=float, default=5.0, help="Polling interval (sec)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    last_n = -1
    while True:
        if csv_path.is_file():
            try:
                df = pd.read_csv(csv_path)
                n = int(len(df))
                if n != last_n:
                    last_n = n
                    tail = df.tail(5)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[WATCH] rows={n} last5:\n")
                        f.write(tail.to_string(index=False))
                        f.write("\n\n")
            except Exception as e:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[WATCH][ERROR] {type(e).__name__}: {e}\n")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
