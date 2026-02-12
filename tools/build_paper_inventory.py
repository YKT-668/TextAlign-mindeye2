#!/usr/bin/env python3
"""Build paper inventory + dashboard numbers (read-only).

Writes to results/paper_inventory/:
- inventory_artifacts.md
- dashboard_numbers.md
- candidate_materials.tsv

This script does NOT run training/inference/evaluation.
It only reads existing CSV/JSON artifacts.

Usage:
  python tools/build_paper_inventory.py
  python tools/build_paper_inventory.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricValue:
    mean: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    n: Optional[int]
    seed: Optional[int]
    bootstrap: Optional[int]
    source: str
    details: Dict[str, Any]


def _read_key_paths(root: Path) -> List[str]:
    p = root / "results/paper_inventory/KEY_PATHS.txt"
    lines: List[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in r]


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _try_int(x: Any) -> Optional[int]:
    v = _try_float(x)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _parse_ci_brackets(s: str) -> Tuple[Optional[float], Optional[float]]:
    # Accept formats like "[0.0804, 0.1192]" or "[0.0804,0.1192]".
    t = (s or "").strip()
    if not t:
        return None, None
    m = re.match(r"^\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]$", t)
    if not m:
        return None, None
    return _try_float(m.group(1)), _try_float(m.group(2))


def _ci_overlap(a: MetricValue, b: MetricValue) -> Optional[bool]:
    if a.ci_lo is None or a.ci_hi is None or b.ci_lo is None or b.ci_hi is None:
        return None
    return not (a.ci_hi < b.ci_lo or b.ci_hi < a.ci_lo)


def _strength(delta: Optional[float], a: MetricValue, b: MetricValue) -> str:
    # b - a
    if delta is None:
        return "UNKNOWN"

    overlap = _ci_overlap(a, b)
    if delta <= 0:
        return "WEAK/NEG"

    if overlap is False:
        return "STRONG"

    if overlap is True:
        return "MED"

    # No CI available: heuristic threshold (5pp)
    return "STRONG" if delta >= 0.05 else "MED"


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def _fmt_ci(v: MetricValue, nd: int = 4) -> str:
    if v.ci_lo is None or v.ci_hi is None:
        return "NA"
    return f"[{v.ci_lo:.{nd}f}, {v.ci_hi:.{nd}f}]"


def _pick_rows(rows: List[Dict[str, str]], **eq: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        ok = True
        for k, v in eq.items():
            if str(r.get(k, "")).strip() != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _module_from_path(rel: str) -> str:
    p = rel.replace("\\", "/")
    if "main_results.csv" in p:
        return "meta_reports"
    if "retrieval" in p.lower():
        return "L1_retrieval"
    if "twoafc" in p.lower():
        return "L1_twoafc"
    if "isrsa" in p.lower():
        return "B_isrsa"
    if "efficiency" in p.lower():
        return "A_efficiency"
    if "ccd_ablation" in p.lower():
        return "C_ablation"
    if "ccd_by_type" in p.lower() or "by_type" in p.lower():
        return "L2_ccd_bytype"
    if "/audit" in p.lower():
        return "L2_ccd_audit"
    if "shared982_ccd" in p.lower() or "ccd_" in p.lower():
        return "L2_ccd"
    if "rsa" in p.lower():
        return "L3_rsa"
    return "neutral"


def _load_manifest_png_names(root: Path, rel_manifest: str) -> Tuple[List[str], str]:
    p = root / rel_manifest
    if not p.exists():
        return [], f"missing: {rel_manifest}"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return [], f"failed to parse json: {type(e).__name__}: {e}"

    names: List[str] = []

    # Supported manifest schemas seen in this repo:
    # 1) list[dict]: each item has path/file/png/relpath/name
    # 2) dict: {figures: list[dict]} with the same keys
    # 3) dict: {figures: {"Fig_x.png": {..., out_path: "results/.../Fig_x.png"}}}
    def add_name(x: Any) -> None:
        if not x:
            return
        try:
            names.append(Path(str(x)).name)
        except Exception:
            return

    def handle_item(it: Dict[str, Any]) -> None:
        add_name(it.get("path") or it.get("file") or it.get("png") or it.get("relpath") or it.get("name"))

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                handle_item(x)
    elif isinstance(data, dict):
        cand = data.get("figures") or data.get("items") or []
        if isinstance(cand, list):
            for x in cand:
                if isinstance(x, dict):
                    handle_item(x)
        elif isinstance(cand, dict):
            # keys are figure names
            for k, v in cand.items():
                add_name(k)
                if isinstance(v, dict):
                    # some manifests provide a concrete output path
                    add_name(v.get("out_path") or v.get("path") or v.get("file"))

    names = sorted(set([n for n in names if n]))
    return names, "ok"


def _metric_from_main_results(
    rows: List[Dict[str, str]],
    *,
    group: str,
    subj: str,
    tag: Optional[str],
    mean_col: str,
    n_col: Optional[str] = None,
    ci_col: Optional[str] = None,
    ci_lo_col: Optional[str] = None,
    ci_hi_col: Optional[str] = None,
    source: str,
) -> Optional[MetricValue]:
    subset = [r for r in rows if str(r.get("group", "")).strip() == group and str(r.get("subj", "")).strip() == subj]
    if tag is not None:
        subset = [r for r in subset if str(r.get("tag", "")).strip() == tag]

    if not subset:
        return None

    # Prefer first match (caller should ensure uniqueness).
    r = subset[0]
    mean = _try_float(r.get(mean_col, ""))
    n = _try_int(r.get(n_col, "")) if n_col else None

    ci_lo = None
    ci_hi = None
    if ci_lo_col and ci_hi_col:
        ci_lo = _try_float(r.get(ci_lo_col, ""))
        ci_hi = _try_float(r.get(ci_hi_col, ""))
    elif ci_col:
        ci_lo, ci_hi = _parse_ci_brackets(str(r.get(ci_col, "")))

    return MetricValue(
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n=n,
        seed=None,
        bootstrap=None,
        source=source,
        details={"group": group, "subj": subj, "tag": str(r.get("tag", "")).strip()},
    )


def _best_ccd_row_for_group(rows: List[Dict[str, str]], group: str, subj: str) -> Optional[Dict[str, str]]:
    # Pick pooled_mean row with max ccd_acc1
    cand = [r for r in rows if str(r.get("group", "")).strip() == group and str(r.get("subj", "")).strip() == subj]
    cand = [r for r in cand if str(r.get("eval_repr", "")).strip() in {"pooled_mean", "pooled"}]
    best: Optional[Dict[str, str]] = None
    best_val: float = -1e9
    for r in cand:
        v = _try_float(r.get("ccd_acc1", ""))
        if v is None:
            continue
        if v > best_val:
            best_val = v
            best = r
    return best


def _ccd_metric(rows: List[Dict[str, str]], group: str, subj: str, source: str) -> Optional[MetricValue]:
    r = _best_ccd_row_for_group(rows, group=group, subj=subj)
    if not r:
        return None

    mean = _try_float(r.get("ccd_acc1", ""))
    ci_lo = _try_float(r.get("ccd_acc1_ci95_lo", ""))
    ci_hi = _try_float(r.get("ccd_acc1_ci95_hi", ""))
    n = _try_int(r.get("n_eval", ""))
    seed = _try_int(r.get("seed", ""))
    bootstrap = _try_int(r.get("bootstrap", ""))

    return MetricValue(
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n=n,
        seed=seed,
        bootstrap=bootstrap,
        source=source,
        details={"group": str(r.get("group", "")).strip(), "subj": subj, "tag": str(r.get("tag", "")).strip(), "eval_repr": str(r.get("eval_repr", "")).strip()},
    )


def _ccd_by_type_metric(rows: List[Dict[str, str]], *, ccd_type: str, group: str, subj: str, source: str) -> Optional[MetricValue]:
    cand = [
        r
        for r in rows
        if str(r.get("type", "")).strip() == ccd_type
        and str(r.get("group", "")).strip() == group
        and str(r.get("subj", "")).strip() == subj
        and str(r.get("eval_repr", "")).strip() in {"pooled_mean", "pooled"}
    ]
    if not cand:
        return None

    # pick max acc1
    best = None
    best_val = -1e9
    for r in cand:
        v = _try_float(r.get("ccd_acc1", ""))
        if v is None:
            continue
        if v > best_val:
            best_val = v
            best = r
    if not best:
        return None

    return MetricValue(
        mean=_try_float(best.get("ccd_acc1", "")),
        ci_lo=_try_float(best.get("ccd_acc1_ci95_lo", "")),
        ci_hi=_try_float(best.get("ccd_acc1_ci95_hi", "")),
        n=_try_int(best.get("n_eval", "")),
        seed=_try_int(best.get("seed", "")),
        bootstrap=_try_int(best.get("bootstrap", "")),
        source=source,
        details={
            "type": ccd_type,
            "group": group,
            "subj": subj,
            "tag": str(best.get("tag", "")).strip(),
            "eval_repr": str(best.get("eval_repr", "")).strip(),
        },
    )


def _write_inventory_artifacts(root: Path, out_dir: Path, key_paths: List[str]) -> None:
    manifest_paths = [
        "cache/model_eval_results/shared982_ccd/figures/figures_manifest.json",
        "cache/model_eval_results/shared982_isrsa/figures/figures_manifest.json",
        "cache/model_eval_results/shared982_efficiency/figures/figures_manifest.json",
    ]

    lines: List[str] = []
    lines.append("# Paper inventory: artifacts")
    lines.append("")
    lines.append("This is a read-only inventory generated from KEY_PATHS.txt and manifests.")
    lines.append("")

    lines.append("## Key paths (from KEY_PATHS.txt)")
    lines.append("")
    for rel in key_paths:
        p = root / rel
        exists = p.exists()
        size = p.stat().st_size if exists else None
        lines.append(f"- {rel} | exists={exists} | size={size}")

    lines.append("")
    lines.append("## Figures manifests")
    lines.append("")
    for rel in manifest_paths:
        names, status = _load_manifest_png_names(root, rel)
        lines.append(f"### {rel}")
        lines.append("")
        lines.append(f"- status: {status}")
        if names:
            lines.append(f"- pngs: {len(names)}")
            for n in names:
                lines.append(f"  - {n}")
        else:
            lines.append("- pngs: 0")
        lines.append("")

    # Meta reports: list symlinked report filenames
    meta_dir = Path("/mnt/work/paper_assets/ALL/meta_reports")
    lines.append("## meta_reports symlinks")
    lines.append("")
    if meta_dir.exists():
        links = sorted([p.name for p in meta_dir.iterdir() if p.is_symlink()])
        lines.append(f"- count: {len(links)}")
        for n in links[:200]:
            lines.append(f"  - {n}")
        if len(links) > 200:
            lines.append(f"  - ... ({len(links)-200} more)")
    else:
        lines.append("- (missing) /mnt/work/paper_assets/ALL/meta_reports")

    out_path = out_dir / "inventory_artifacts.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dashboard_numbers(root: Path, out_dir: Path) -> Dict[str, Any]:
    # Load core tables
    main_results = _read_csv(root / "results/tables/main_results.csv")
    isrsa_summary = _read_csv(root / "results/tables/isrsa_summary.csv")
    efficiency_summary = _read_csv(root / "results/tables/efficiency_summary.csv")
    ccd_summary = _read_csv(root / "cache/model_eval_results/shared982_ccd/ccd_summary.csv")
    ccd_by_type = _read_csv(root / "cache/model_eval_results/shared982_ccd/ccd_by_type.csv")

    # baseline/textalign proxy mapping (because many shared982 artifacts use subject-specific groups)
    proxy = {
        "1": {"baseline": "official_hf", "textalign_llm": "ours_s1_v2"},
        "2": {"baseline": "official_hf", "textalign_llm": "ours_s2_v10"},
        "5": {"baseline": "official_hf", "textalign_llm": "ours_s5_v10"},
        "7": {"baseline": "official_hf", "textalign_llm": "ours_s7_v10"},
    }

    lines: List[str] = []
    lines.append("# Dashboard: key numbers")
    lines.append("")
    lines.append("Notes:")
    lines.append("- This dashboard is read-only (no re-eval).")
    lines.append("- Some shared982 artifacts do not literally use model tags 'baseline'/'textalign_llm'.")
    lines.append("  When absent, we include a PROXY comparison using available group names (documented per row).")
    lines.append("")

    strengths: Dict[str, Dict[str, str]] = {}  # metric_id -> subj -> strength

    def add_strength(metric_id: str, subj: str, s: str) -> None:
        strengths.setdefault(metric_id, {})[subj] = s

    # L1 retrieval (from main_results shared982_retrieval rows; no CI)
    lines.append("## L1 retrieval (Top-1/Top-5)")
    lines.append("")
    retr_rows = [r for r in main_results if str(r.get("group", "")).strip() == "shared982_retrieval"]
    if retr_rows:
        lines.append("Source: results/tables/main_results.csv (group=shared982_retrieval)")
        lines.append("")
        lines.append("| subj | tag | N | fwd@1 | fwd@5 | bwd@1 | bwd@5 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for subj in ["1", "2", "5", "7"]:
            rr = [r for r in retr_rows if str(r.get("subj", "")).strip() == subj]
            if not rr:
                lines.append(f"| {subj} | NA | NA | NA | NA | NA | NA |")
                continue
            r = rr[0]
            lines.append(
                "| {subj} | {tag} | {n} | {f1} | {f5} | {b1} | {b5} |".format(
                    subj=subj,
                    tag=str(r.get("tag", "")).strip() or "NA",
                    n=str(r.get("retrieval_N", "")).strip() or "NA",
                    f1=_fmt(_try_float(r.get("retrieval_fwd_top1"))),
                    f5=_fmt(_try_float(r.get("retrieval_fwd_top5"))),
                    b1=_fmt(_try_float(r.get("retrieval_bwd_top1"))),
                    b5=_fmt(_try_float(r.get("retrieval_bwd_top5"))),
                )
            )
        lines.append("")
        lines.append("Delta (textalign_llm - baseline): NOT AVAILABLE from these rows (no baseline/textalign tags here).")
        lines.append("(See PROXY comparison in CCD/RSA sections below, or use efficiency block for baseline/textalign.)")
    else:
        lines.append("shared982_retrieval rows not found in main_results.csv; cannot summarize L1 retrieval.")
    lines.append("")

    # L1 twoafc (from main_results shared982_twoafc)
    lines.append("## L1 twoafc")
    lines.append("")
    two_rows = [r for r in main_results if str(r.get("group", "")).strip() == "shared982_twoafc"]
    if two_rows:
        lines.append("Source: results/tables/main_results.csv (group=shared982_twoafc)")
        lines.append("")
        lines.append("| subj | tag | N | twoafc_fwd |")
        lines.append("|---:|---|---:|---:|")
        for subj in ["1", "2", "5", "7"]:
            rr = [r for r in two_rows if str(r.get("subj", "")).strip() == subj]
            if not rr:
                lines.append(f"| {subj} | NA | NA | NA |")
                continue
            r = rr[0]
            lines.append(
                "| {subj} | {tag} | {n} | {v} |".format(
                    subj=subj,
                    tag=str(r.get("tag", "")).strip() or "NA",
                    n=str(r.get("retrieval_N", "")).strip() or "NA",
                    v=_fmt(_try_float(r.get("twoafc_fwd"))),
                )
            )
        lines.append("")
        lines.append("Delta (textalign_llm - baseline): NOT AVAILABLE from these rows (no baseline/textalign tags here).")
    else:
        lines.append("shared982_twoafc rows not found in main_results.csv; cannot summarize L1 twoafc.")
    lines.append("")

    # L3 RSA (proxy from main_results rsa_pearson using official_hf vs ours_s* tags)
    lines.append("## L3 RSA (proxy: official_hf vs ours_s*)")
    lines.append("")
    lines.append("Source: results/tables/main_results.csv (rsa_pearson column; proxy mapping per subject)")
    lines.append("")
    lines.append("| subj | baseline(group/tag) | rho | textalign(group/tag) | rho | delta | strength |")
    lines.append("|---:|---|---:|---|---:|---:|---|")
    for subj in ["1", "2", "5", "7"]:
        pb = proxy[subj]["baseline"]
        pt = proxy[subj]["textalign_llm"]

        # pick baseline row with pooled metrics (tag contains 40sess when possible)
        b_rows = [r for r in main_results if str(r.get("group", "")).strip() == pb and str(r.get("subj", "")).strip() == subj and str(r.get("rsa_pearson", "")).strip()]
        b_rows = sorted(b_rows, key=lambda r: ("40sess" not in str(r.get("tag", "")), -(_try_float(r.get("rsa_pearson")) or -1e9)))
        t_rows = [r for r in main_results if str(r.get("group", "")).strip() == pt and str(r.get("subj", "")).strip() == subj and str(r.get("rsa_pearson", "")).strip()]
        t_rows = sorted(t_rows, key=lambda r: ("40sess" not in str(r.get("tag", "")), -(_try_float(r.get("rsa_pearson")) or -1e9)))

        b = b_rows[0] if b_rows else None
        t = t_rows[0] if t_rows else None

        b_mv = MetricValue(_try_float(b.get("rsa_pearson")) if b else None, None, None, None, None, None, "main_results.csv", {"group": pb, "tag": (b.get("tag") if b else None)})
        t_mv = MetricValue(_try_float(t.get("rsa_pearson")) if t else None, None, None, None, None, None, "main_results.csv", {"group": pt, "tag": (t.get("tag") if t else None)})
        delta = (t_mv.mean - b_mv.mean) if (t_mv.mean is not None and b_mv.mean is not None) else None
        st = _strength(delta, b_mv, t_mv)
        add_strength("rsa_proxy", subj, st)
        lines.append(
            f"| {subj} | {pb}/{(b.get('tag','NA') if b else 'NA')} | {_fmt(b_mv.mean)} | {pt}/{(t.get('tag','NA') if t else 'NA')} | {_fmt(t_mv.mean)} | {_fmt(delta)} | {st} |"
        )
    lines.append("")

    # L2 CCD hardneg (proxy official_hf vs ours_s*)
    lines.append("## L2 CCD hardneg (proxy: official_hf vs ours_s*)")
    lines.append("")
    lines.append("Source: cache/model_eval_results/shared982_ccd/ccd_summary.csv (pooled_mean rows)")
    lines.append("")
    lines.append("| subj | baseline(group/tag) | acc1 | CI95 | N | textalign(group/tag) | acc1 | CI95 | N | delta | strength |")
    lines.append("|---:|---|---:|---|---:|---|---:|---|---:|---:|---|")
    for subj in ["1", "2", "5", "7"]:
        pb = proxy[subj]["baseline"]
        pt = proxy[subj]["textalign_llm"]
        bmv = _ccd_metric(ccd_summary, pb, subj.zfill(2), "ccd_summary.csv")
        tmv = _ccd_metric(ccd_summary, pt, subj.zfill(2), "ccd_summary.csv")
        delta = (tmv.mean - bmv.mean) if (tmv and bmv and tmv.mean is not None and bmv.mean is not None) else None
        st = _strength(delta, bmv or MetricValue(None, None, None, None, None, None, "", {}), tmv or MetricValue(None, None, None, None, None, None, "", {}))
        add_strength("ccd_proxy", subj, st)
        lines.append(
            "| {subj} | {bg}/{bt} | {b} | {bci} | {bn} | {tg}/{tt} | {t} | {tci} | {tn} | {d} | {st} |".format(
                subj=subj,
                bg=(bmv.details.get("group") if bmv else pb),
                bt=(bmv.details.get("tag") if bmv else "NA"),
                b=_fmt(bmv.mean if bmv else None),
                bci=_fmt_ci(bmv) if bmv else "NA",
                bn=str(bmv.n) if bmv and bmv.n is not None else "NA",
                tg=(tmv.details.get("group") if tmv else pt),
                tt=(tmv.details.get("tag") if tmv else "NA"),
                t=_fmt(tmv.mean if tmv else None),
                tci=_fmt_ci(tmv) if tmv else "NA",
                tn=str(tmv.n) if tmv and tmv.n is not None else "NA",
                d=_fmt(delta),
                st=st,
            )
        )
    lines.append("")

    # L2 CCD by-type (proxy)
    lines.append("## L2 CCD by-type (proxy: official_hf vs ours_s*)")
    lines.append("")
    lines.append("Source: cache/model_eval_results/shared982_ccd/ccd_by_type.csv")
    lines.append("")
    for ccd_type in ["object", "attribute", "relation"]:
        lines.append(f"### type={ccd_type}")
        lines.append("")
        lines.append("| subj | baseline acc1 | CI95 | N | textalign acc1 | CI95 | N | delta | strength |")
        lines.append("|---:|---:|---|---:|---:|---|---:|---:|---|")
        for subj in ["1", "2", "5", "7"]:
            pb = proxy[subj]["baseline"]
            pt = proxy[subj]["textalign_llm"]
            bmv = _ccd_by_type_metric(ccd_by_type, ccd_type=ccd_type, group=pb, subj=subj, source="ccd_by_type.csv")
            tmv = _ccd_by_type_metric(ccd_by_type, ccd_type=ccd_type, group=pt, subj=subj, source="ccd_by_type.csv")
            delta = (tmv.mean - bmv.mean) if (tmv and bmv and tmv.mean is not None and bmv.mean is not None) else None
            st = _strength(delta, bmv or MetricValue(None, None, None, None, None, None, "", {}), tmv or MetricValue(None, None, None, None, None, None, "", {}))
            add_strength(f"ccd_by_type_{ccd_type}_proxy", subj, st)
            lines.append(
                "| {subj} | {b} | {bci} | {bn} | {t} | {tci} | {tn} | {d} | {st} |".format(
                    subj=subj,
                    b=_fmt(bmv.mean if bmv else None),
                    bci=_fmt_ci(bmv) if bmv else "NA",
                    bn=str(bmv.n) if bmv and bmv.n is not None else "NA",
                    t=_fmt(tmv.mean if tmv else None),
                    tci=_fmt_ci(tmv) if tmv else "NA",
                    tn=str(tmv.n) if tmv and tmv.n is not None else "NA",
                    d=_fmt(delta),
                    st=st,
                )
            )
        lines.append("")

    # B IS-RSA (exact baseline vs textalign_llm)
    lines.append("## B IS-RSA (baseline vs textalign_llm)")
    lines.append("")
    b = _pick_rows(isrsa_summary, model_tag="baseline")
    t = _pick_rows(isrsa_summary, model_tag="textalign_llm")
    if b and t:
        bmv = MetricValue(
            mean=_try_float(b[0].get("mean_offdiag_isrsa")),
            ci_lo=_try_float(b[0].get("mean_offdiag_isrsa_ci95_lo")),
            ci_hi=_try_float(b[0].get("mean_offdiag_isrsa_ci95_hi")),
            n=_try_int(b[0].get("N")),
            seed=None,
            bootstrap=None,
            source="isrsa_summary.csv",
            details={"model_tag": "baseline"},
        )
        tmv = MetricValue(
            mean=_try_float(t[0].get("mean_offdiag_isrsa")),
            ci_lo=_try_float(t[0].get("mean_offdiag_isrsa_ci95_lo")),
            ci_hi=_try_float(t[0].get("mean_offdiag_isrsa_ci95_hi")),
            n=_try_int(t[0].get("N")),
            seed=None,
            bootstrap=None,
            source="isrsa_summary.csv",
            details={"model_tag": "textalign_llm"},
        )
        delta = (tmv.mean - bmv.mean) if (tmv.mean is not None and bmv.mean is not None) else None
        st = _strength(delta, bmv, tmv)
        add_strength("isrsa", "all", st)
        lines.append("| metric | baseline | CI95 | N | textalign_llm | CI95 | N | delta | strength |")
        lines.append("|---|---:|---|---:|---:|---|---:|---:|---|")
        lines.append(
            f"| mean_offdiag_isrsa | {_fmt(bmv.mean)} | {_fmt_ci(bmv)} | {bmv.n} | {_fmt(tmv.mean)} | {_fmt_ci(tmv)} | {tmv.n} | {_fmt(delta)} | {st} |"
        )
    else:
        lines.append("Missing baseline/textalign_llm rows in isrsa_summary.csv")
    lines.append("")

    # A efficiency (exact baseline vs textalign_llm; subj=1/5; settings 1/2/40)
    lines.append("## A efficiency (baseline vs textalign_llm; subj=1/5; 1/2/40 sess)")
    lines.append("")

    def eff_metric(model: str, subj: str, setting: str, col_mean: str, col_lo: str, col_hi: str, metric_id: str) -> Optional[MetricValue]:
        rr = [
            r
            for r in efficiency_summary
            if str(r.get("model", "")).strip() == model
            and str(r.get("subj", "")).strip() == subj
            and str(r.get("setting", "")).strip().lower() == setting
        ]
        if not rr:
            return None
        r = rr[0]
        return MetricValue(
            mean=_try_float(r.get(col_mean)),
            ci_lo=_try_float(r.get(col_lo)),
            ci_hi=_try_float(r.get(col_hi)),
            n=_try_int(r.get("N")),
            seed=_try_int(r.get("seed")),
            bootstrap=None,
            source="efficiency_summary.csv",
            details={"subj": subj, "model": model, "setting": setting, "metric": metric_id},
        )

    for metric_id, cols in [
        ("ccd_acc1", ("ccd_acc1", "ccd_ci_lo", "ccd_ci_hi")),
        ("twoafc_hard", ("twoafc_hard", "twoafc_ci_lo", "twoafc_ci_hi")),
        ("rsa_rho", ("rsa_rho", "rsa_ci_lo", "rsa_ci_hi")),
    ]:
        lines.append(f"### {metric_id}")
        lines.append("")
        lines.append("| subj | setting | baseline | CI95 | textalign_llm | CI95 | delta | strength |")
        lines.append("|---:|---|---:|---|---:|---|---:|---|")
        for subj in ["1", "5"]:
            for setting in ["1sess", "2sess", "40sess"]:
                bmv = eff_metric("baseline", subj, setting, cols[0], cols[1], cols[2], metric_id)
                tmv = eff_metric("textalign_llm", subj, setting, cols[0], cols[1], cols[2], metric_id)
                delta = (tmv.mean - bmv.mean) if (tmv and bmv and tmv.mean is not None and bmv.mean is not None) else None
                st = _strength(delta, bmv or MetricValue(None, None, None, None, None, None, "", {}), tmv or MetricValue(None, None, None, None, None, None, "", {}))
                add_strength(f"eff_{metric_id}", f"{subj}:{setting}", st)
                lines.append(
                    "| {subj} | {setting} | {b} | {bci} | {t} | {tci} | {d} | {st} |".format(
                        subj=subj,
                        setting=setting,
                        b=_fmt(bmv.mean if bmv else None),
                        bci=_fmt_ci(bmv) if bmv else "NA",
                        t=_fmt(tmv.mean if tmv else None),
                        tci=_fmt_ci(tmv) if tmv else "NA",
                        d=_fmt(delta),
                        st=st,
                    )
                )
        lines.append("")

    out_path = out_dir / "dashboard_numbers.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # return summary for candidate_materials
    return {"strengths": strengths}


def _write_candidate_materials(root: Path, out_dir: Path, key_paths: List[str], dashboard_meta: Dict[str, Any]) -> None:
    strengths: Dict[str, Dict[str, str]] = dashboard_meta.get("strengths", {})

    def module_strength(module: str) -> str:
        # If any STRONG in that module's metrics -> STRONG else MED/WEAK.
        flat: List[str] = []
        for metric_id, by_subj in strengths.items():
            if module == "L2_ccd" and metric_id.startswith("ccd"):
                flat.extend(by_subj.values())
            elif module == "L2_ccd_bytype" and metric_id.startswith("ccd_by_type"):
                flat.extend(by_subj.values())
            elif module == "L3_rsa" and metric_id.startswith("rsa"):
                flat.extend(by_subj.values())
            elif module == "B_isrsa" and metric_id == "isrsa":
                flat.extend(by_subj.values())
            elif module == "A_efficiency" and metric_id.startswith("eff_"):
                flat.extend(by_subj.values())
        if any(s == "STRONG" for s in flat):
            return "STRONG"
        if any(s == "MED" for s in flat):
            return "MED"
        if any(s == "WEAK/NEG" for s in flat):
            return "WEAK/NEG"
        return "UNKNOWN"

    def bucket_for_module(module: str) -> str:
        s = module_strength(module)
        return "pre_main" if s == "STRONG" else "pre_supp"

    rows: List[List[str]] = []

    # 1) Key paths as tables/figs
    for rel in key_paths:
        p = root / rel
        if not p.exists():
            continue
        module = _module_from_path(rel)
        typ = "table" if rel.lower().endswith((".csv", ".tsv")) else ("fig" if rel.lower().endswith(".png") else "report")
        name = Path(rel).name
        bucket = bucket_for_module(module)
        reason = f"key_path ({module_strength(module)})"
        rows.append([module, typ, name, rel, bucket, reason])

    # 2) Figures from manifests
    manifest_targets = [
        ("cache/model_eval_results/shared982_ccd/figures/figures_manifest.json", "L2_ccd"),
        ("cache/model_eval_results/shared982_isrsa/figures/figures_manifest.json", "B_isrsa"),
        ("cache/model_eval_results/shared982_efficiency/figures/figures_manifest.json", "A_efficiency"),
    ]
    for rel_m, module in manifest_targets:
        names, status = _load_manifest_png_names(root, rel_m)
        if status != "ok":
            continue
        # Path rel: we keep the manifest-relative png names only (as requested).
        for n in names:
            rows.append([module, "fig", n, "(from manifest) " + rel_m, bucket_for_module(module), f"manifest ({module_strength(module)})"])

    out_path = out_dir / "candidate_materials.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["module", "type", "name", "path_rel", "suggested_bucket", "reason_short"])
        for r in rows:
            w.writerow(r)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Build paper inventory artifacts + dashboard (read-only)")
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1] if args.root is None else Path(args.root).expanduser().resolve()
    out_dir = root / "results/paper_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_paths = _read_key_paths(root)

    _write_inventory_artifacts(root, out_dir, key_paths)
    dashboard_meta = _write_dashboard_numbers(root, out_dir)
    _write_candidate_materials(root, out_dir, key_paths, dashboard_meta)

    print("WROTE:", str(out_dir / "inventory_artifacts.md"))
    print("WROTE:", str(out_dir / "dashboard_numbers.md"))
    print("WROTE:", str(out_dir / "candidate_materials.tsv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
