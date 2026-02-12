#!/usr/bin/env python
# coding: utf-8
"""watch_official_hf_progress.py

Continuously watches official HF sweep progress.

- Reads the summary CSV and prints/records any newly appended rows.
- Focuses on pooled rows: prints `tag + pooled FWD@1/BWD@1`.
- Also prints a short tail of the running sweep log each iteration.

This is intentionally lightweight and dependency-free.
"""

import argparse
import csv
import os
import time
from pathlib import Path


def _tail_lines(path: Path, n: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "(log not found yet)"
    lines = data.replace("\r", "\n").splitlines()
    return "\n".join(lines[-n:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="cache/model_eval_results/official_hf_baselines_summary.csv",
        help="Path to the sweep summary CSV",
    )
    ap.add_argument(
        "--log",
        default="cache/model_eval_results/official_hf/official_hf_sweep.log",
        help="Path to the running sweep log",
    )
    ap.add_argument(
        "--out",
        default="cache/model_eval_results/official_hf/official_hf_progress_updates.log",
        help="Where to write watcher updates",
    )
    ap.add_argument("--interval", type=float, default=30.0, help="Polling interval (seconds)")
    ap.add_argument("--tail", type=int, default=25, help="How many log lines to include each tick")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_seen_rows = 0

    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append(f"\n===== {stamp} =====")

        # Read CSV and detect new rows
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            if len(rows) > last_seen_rows:
                new_rows = rows[last_seen_rows:]
                last_seen_rows = len(rows)

                pooled = [r for r in new_rows if r.get("eval_repr") == "pooled"]
                if pooled:
                    lines.append("[NEW pooled results]")
                    for r in pooled:
                        try:
                            subj = int(r["subj"])
                            tag = r["tag"]
                            fwd1 = float(r["fwd_top1"]) * 100.0
                            bwd1 = float(r["bwd_top1"]) * 100.0
                            lines.append(f"subj{subj:02d}\t{tag}\tpooled\tFWD@1={fwd1:.2f}%\tBWD@1={bwd1:.2f}%")
                        except Exception:
                            lines.append(f"(failed to parse row) {r}")
                else:
                    lines.append(f"[NEW rows] +{len(new_rows)} (no pooled rows in the new chunk)")
            else:
                lines.append(f"[CSV] rows={len(rows)} (no new rows)")
        else:
            lines.append("[CSV] not found yet")

        # Tail sweep log
        lines.append("[LOG tail]")
        lines.append(_tail_lines(log_path, args.tail))

        msg = "\n".join(lines) + "\n"
        print(msg, flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(msg)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
