#!/usr/bin/env python
# coding: utf-8
"""build_main_results_table.py

Merge key evaluation products (L1/retrieval + 2AFC + RSA + CCD) into one main table.

Writes:
- /mnt/work/results/tables/main_results.csv

By default, focuses on pooled representations:
- retrieval: eval_repr == pooled
- twoafc: eval_repr == pooled
- rsa: eval_repr in {pooled, pooled_mean}
- ccd: eval_repr == pooled_mean

This keeps the final table compact and paper-friendly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_csv",
        default="/mnt/work/results/tables/main_results.csv",
    )
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    retrieval_csv = PROJ / "cache" / "model_eval_results" / "shared982" / "retrieval_summary.csv"
    twoafc_csv = PROJ / "cache" / "model_eval_results" / "shared982_twoafc" / "twoafc_summary.csv"
    rsa_csv = PROJ / "cache" / "model_eval_results" / "shared982_rsa" / "rsa_summary.csv"
    ccd_csv = PROJ / "cache" / "model_eval_results" / "shared982_ccd" / "ccd_summary.csv"

    if not retrieval_csv.is_file():
        raise RuntimeError(f"Missing {retrieval_csv}")
    if not twoafc_csv.is_file():
        raise RuntimeError(f"Missing {twoafc_csv}")
    if not rsa_csv.is_file():
        raise RuntimeError(f"Missing {rsa_csv}")
    if not ccd_csv.is_file():
        raise RuntimeError(f"Missing {ccd_csv}")

    r = pd.read_csv(retrieval_csv)
    r = r[r["eval_repr"] == "pooled"].copy()
    r = r.rename(
        columns={
            "N": "retrieval_N",
            "fwd_top1": "retrieval_fwd_top1",
            "fwd_top5": "retrieval_fwd_top5",
            "bwd_top1": "retrieval_bwd_top1",
            "bwd_top5": "retrieval_bwd_top5",
        }
    )
    r = r[["group", "tag", "subj", "retrieval_N", "retrieval_fwd_top1", "retrieval_fwd_top5", "retrieval_bwd_top1", "retrieval_bwd_top5"]]

    t = pd.read_csv(twoafc_csv)
    t = t[t["eval_repr"] == "pooled"].copy()
    # prefer forward mean
    if "twoafc_fwd_mean" in t.columns:
        t = t.rename(columns={"twoafc_fwd_mean": "twoafc_fwd"})
    t = t[["group", "tag", "subj", "twoafc_fwd"]]

    s = pd.read_csv(rsa_csv)
    if "eval_repr" in s.columns:
        s = s[s["eval_repr"].isin(["pooled", "pooled_mean"])].copy()
    s = s[["group", "tag", "subj", "rsa_pearson"]]

    c = pd.read_csv(ccd_csv)
    c = c[c["eval_repr"] == "pooled_mean"].copy()
    # add compact CI text columns
    if "ccd_acc1_ci95_lo" in c.columns and "ccd_acc1_ci95_hi" in c.columns:
        def _ci_str(lo, hi):
            if lo == "" or hi == "" or pd.isna(lo) or pd.isna(hi):
                return ""
            return f"[{float(lo):.4f}, {float(hi):.4f}]"

        c["ccd_acc1_ci95"] = [_ci_str(lo, hi) for lo, hi in zip(c["ccd_acc1_ci95_lo"], c["ccd_acc1_ci95_hi"]) ]
    c = c[["group", "tag", "subj", "neg_mode", "n_eval", "ccd_acc1", "ccd_acc1_ci95", "twoafc_hardest", "margin_mean"]]
    c = c.rename(columns={"n_eval": "ccd_N"})

    # merge
    j = r.merge(t, on=["group", "tag", "subj"], how="outer")
    j = j.merge(s, on=["group", "tag", "subj"], how="outer")
    j = j.merge(c, on=["group", "tag", "subj"], how="outer")

    # sort for readability
    j["subj"] = j["subj"].astype(str).str.zfill(2)
    j = j.sort_values(["subj", "group", "tag"]).reset_index(drop=True)

    j.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()
