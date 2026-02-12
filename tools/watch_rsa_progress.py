#!/usr/bin/env python
# coding: utf-8

"""Simple watcher for RSA batch progress.

It polls rsa_summary.csv and prints newly added rows.
Usage:
  python tools/watch_rsa_progress.py
Then in another terminal:
  python tools/rerun_all_rsa_shared982.py

Or:
  tail -f cache/model_eval_results/shared982_rsa/rsa_watcher.out
"""

from __future__ import annotations

import csv
import time
from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]
CSV_PATH = PROJ / "cache" / "model_eval_results" / "shared982_rsa" / "rsa_summary.csv"
OUT_PATH = PROJ / "cache" / "model_eval_results" / "shared982_rsa" / "rsa_watcher.out"


def main() -> None:
    seen = set()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("", encoding="utf-8")
    print(f"[WATCH] polling {CSV_PATH}")

    while True:
        if CSV_PATH.is_file():
            with CSV_PATH.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    key = (row.get("group", ""), row.get("tag", ""), row.get("eval_repr", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    line = (
                        f"{row.get('group','')}\t{row.get('tag','')}\t{row.get('eval_repr','')}\t"
                        f"spearman={row.get('rsa_spearman','')}\tpearson={row.get('rsa_pearson','')}\n"
                    )
                    with OUT_PATH.open("a", encoding="utf-8") as o:
                        o.write(line)
                    print(line, end="")
        time.sleep(2.0)


if __name__ == "__main__":
    main()
