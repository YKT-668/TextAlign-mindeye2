#!/usr/bin/env python3
"""Audit final experimental artifacts (read-only).

This script only checks presence/format/consistency of expected outputs.
It does NOT run training/inference/evaluation.

Outputs:
- results/ARTIFACT_AUDIT.md
- results/ARTIFACT_AUDIT.json
- results/ARTIFACT_MISSING.csv (only if missing entries exist)

Usage:
  python tools/audit_all_artifacts.py
  python tools/audit_all_artifacts.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class MissingRow:
    item_id: str
    severity: str
    expected_path: str
    note: str


def _utc_iso(ts: float) -> str:
    return _dt.datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"


def _safe_stat(path: Path) -> Dict[str, Any]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
        "mtime_iso": _utc_iso(st.st_mtime),
    }


def _read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _read_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
            if max_rows is not None and i + 1 >= max_rows:
                break
    return rows


def _csv_nonempty_data_lines(path: Path) -> int:
    # Count non-empty lines excluding header.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _status_merge(statuses: Iterable[str]) -> str:
    # FAIL > WARN > PASS
    statuses = list(statuses)
    if any(s == STATUS_FAIL for s in statuses):
        return STATUS_FAIL
    if any(s == STATUS_WARN for s in statuses):
        return STATUS_WARN
    return STATUS_PASS


def _check_file(
    root: Path,
    rel: str,
    *,
    min_size: int = 1,
    note_if_ok: Optional[str] = None,
    missing: Optional[List[MissingRow]] = None,
    item_id: Optional[str] = None,
    severity: str = "CRITICAL",
    note_if_missing: str = "missing",
) -> Tuple[str, Dict[str, Any]]:
    path = root / rel
    st = _safe_stat(path)
    if not st.get("exists", False):
        if missing is not None and item_id is not None:
            missing.append(
                MissingRow(item_id=item_id, severity=severity, expected_path=rel, note=note_if_missing)
            )
        return STATUS_FAIL, {**st, "notes": note_if_missing}
    if st.get("size_bytes", 0) < min_size:
        if missing is not None and item_id is not None:
            missing.append(
                MissingRow(
                    item_id=item_id,
                    severity=severity,
                    expected_path=rel,
                    note=f"file too small (<{min_size} bytes)",
                )
            )
        return STATUS_FAIL, {**st, "notes": f"file too small (<{min_size} bytes)"}
    return STATUS_PASS, {**st, "notes": note_if_ok or "ok"}


def _list_csvs_under(root: Path, rel_dir: str) -> List[str]:
    base = root / rel_dir
    if not base.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in base.rglob("*.csv") if p.is_file())


def _find_figs_matching(root: Path, rel_dirs: Sequence[str], pattern: str) -> List[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    out: List[str] = []
    for rel_dir in rel_dirs:
        base = root / rel_dir
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if rx.search(p.name):
                out.append(str(p.relative_to(root)))
    return sorted(set(out))


def _parse_float_maybe(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _audit_main_results_metric_presence(
    rows: List[Dict[str, str]],
    *,
    subjs_required: Sequence[int],
    metric_cols_any: Sequence[str],
    group_keyword: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    present_subjs: set[int] = set()
    keyword_rows = 0
    metric_rows = 0

    for r in rows:
        subj = r.get("subj", "")
        subj_i = None
        try:
            subj_i = int(float(subj))
        except Exception:
            subj_i = None

        if group_keyword is not None:
            if group_keyword.lower() in (r.get("group", "").lower()):
                keyword_rows += 1

        if any((r.get(c, "").strip() != "") for c in metric_cols_any):
            metric_rows += 1
            if subj_i is not None:
                present_subjs.add(subj_i)

    missing_subjs = [s for s in subjs_required if s not in present_subjs]

    # Strict expectation: group keyword rows exist; but repo may store metric columns without keyworded groups.
    status = STATUS_PASS
    notes: List[str] = []

    if missing_subjs:
        status = STATUS_FAIL
        notes.append(f"missing required subjects: {missing_subjs}")

    if group_keyword is not None and keyword_rows == 0:
        # If metrics exist but keyworded group rows don't, treat as WARN.
        if metric_rows > 0 and status != STATUS_FAIL:
            status = STATUS_WARN
            notes.append(
                f"no group rows containing '{group_keyword}' found; using non-empty metric columns as evidence"
            )
        elif status != STATUS_FAIL:
            status = STATUS_FAIL
            notes.append(f"no group rows containing '{group_keyword}' and no metric rows found")

    if metric_rows == 0 and status != STATUS_FAIL:
        status = STATUS_FAIL
        notes.append("no non-empty metric rows found")

    return status, {
        "present_subjs": sorted(present_subjs),
        "missing_subjs": missing_subjs,
        "keyword_rows": keyword_rows,
        "metric_rows": metric_rows,
        "notes": "; ".join(notes) if notes else "ok",
    }


def _audit_duplicates_main_results(rows: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    keys_seen: set[Tuple[str, str, str]] = set()
    dup_keys: Dict[Tuple[str, str, str], int] = {}

    for r in rows:
        key = (r.get("group", ""), r.get("subj", ""), r.get("tag", ""))
        if key in keys_seen:
            dup_keys[key] = dup_keys.get(key, 1) + 1
        else:
            keys_seen.add(key)

    if dup_keys:
        return STATUS_WARN, {
            "duplicate_key_count": len(dup_keys),
            "example_duplicates": [
                {"group": k[0], "subj": k[1], "tag": k[2], "count": c} for k, c in list(dup_keys.items())[:10]
            ],
        }
    return STATUS_PASS, {"duplicate_key_count": 0}


def _extract_int_fields_from_json(obj: Any, keys: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                try:
                    out[k] = int(obj[k])
                except Exception:
                    pass
    return out


def _maybe_import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def _npy_feature_dim(path: Path) -> Tuple[Optional[int], str]:
    np = _maybe_import_numpy()
    if np is None:
        return None, "numpy not available; cannot load npy"
    try:
        arr = np.load(str(path), mmap_mode="r")
        shape = getattr(arr, "shape", None)
        if not shape:
            return None, "unable to read shape"
        if len(shape) == 1:
            return int(shape[0]), "1D array"
        return int(shape[-1]), f"shape={shape}"
    except FileNotFoundError:
        return None, "file missing"
    except Exception as e:
        return None, f"failed to load npy: {e}"


def run_audit(
    root: Path,
    *,
    out_md: Optional[Path] = None,
    out_json: Optional[Path] = None,
    out_missing_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    missing: List[MissingRow] = []
    items_out: List[Dict[str, Any]] = []

    # Preload main_results rows once.
    main_results_rel = "results/tables/main_results.csv"
    main_results_path = root / main_results_rel
    main_rows: List[Dict[str, str]] = []
    if main_results_path.exists():
        try:
            main_rows = _read_csv_rows(main_results_path)
        except Exception:
            main_rows = []

    # Helper to register an item.
    def add_item(
        *,
        item_id: str,
        title: str,
        severity: str,
        status: str,
        paths: List[Dict[str, Any]],
        notes: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "item_id": item_id,
            "title": title,
            "severity": severity,
            "status": status,
            "paths": paths,
            "notes": notes,
        }
        if extra:
            payload.update(extra)
        items_out.append(payload)

    # 1. Bi-directional retrieval
    paths: List[Dict[str, Any]] = []
    st_main, p_main = _check_file(
        root,
        main_results_rel,
        min_size=10,
        missing=missing,
        item_id="L1-1",
        severity="CRITICAL",
        note_if_missing="main_results.csv missing",
    )
    paths.append(p_main)

    status = st_main
    notes = ""
    evidence: Dict[str, Any] = {}
    if st_main != STATUS_FAIL and main_rows:
        st_metric, evidence = _audit_main_results_metric_presence(
            main_rows,
            subjs_required=[1, 2, 5, 7],
            metric_cols_any=[
                "retrieval_fwd_top1",
                "retrieval_fwd_top5",
                "retrieval_bwd_top1",
                "retrieval_bwd_top5",
            ],
            group_keyword="retrieval",
        )
        status = _status_merge([status, st_metric])
        notes = evidence.get("notes", "")
    else:
        notes = p_main.get("notes", "")

    table_csvs = _list_csvs_under(root, "results/tables")
    retrieval_related = [p for p in table_csvs if re.search(r"retrieval", Path(p).name, re.IGNORECASE)]
    if retrieval_related:
        notes = (notes + "; " if notes and notes != "ok" else notes) + f"extra csvs: {retrieval_related}"

    add_item(
        item_id="L1-1",
        title="Bi-directional retrieval（Top-1/Top-5）",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes=notes or "ok",
        extra={"details": evidence, "tables_csvs": table_csvs},
    )

    # 2. 2AFC Two-way identification
    paths = []
    paths.append(_safe_stat(main_results_path))
    status = STATUS_PASS if main_results_path.exists() else STATUS_FAIL
    evidence = {}
    notes = ""
    if status != STATUS_FAIL and main_rows:
        st_metric, evidence = _audit_main_results_metric_presence(
            main_rows,
            subjs_required=[1, 2, 5, 7],
            metric_cols_any=["twoafc_fwd"],
            group_keyword="twoafc",
        )
        status = _status_merge([status, st_metric])
        notes = evidence.get("notes", "")

    twoafc_tables = [p for p in table_csvs if re.search(r"twoafc", Path(p).name, re.IGNORECASE)]
    if twoafc_tables:
        notes = (notes + "; " if notes and notes != "ok" else notes) + f"extra csvs: {twoafc_tables}"

    add_item(
        item_id="L1-2",
        title="2AFC Two-way identification",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes=notes or "ok",
        extra={"details": evidence, "twoafc_tables": twoafc_tables},
    )

    # 3. RSA summary
    paths = []
    paths.append(_safe_stat(main_results_path))
    status = STATUS_PASS if main_results_path.exists() else STATUS_FAIL
    evidence = {}
    notes = ""

    if status != STATUS_FAIL and main_rows:
        st_metric, evidence = _audit_main_results_metric_presence(
            main_rows,
            subjs_required=[1, 2, 5, 7],
            metric_cols_any=["rsa_pearson"],
            group_keyword="rsa",
        )
        status = _status_merge([status, st_metric])
        notes = evidence.get("notes", "")

    rsa_figs = _find_figs_matching(
        root,
        ["results/figures_main", "cache/model_eval_results"],
        r"rsa",
    )
    if not rsa_figs:
        status = _status_merge([status, STATUS_FAIL])
        missing.append(
            MissingRow(
                item_id="L3-3",
                severity="CRITICAL",
                expected_path="results/figures_main (or cache/model_eval_results/**/figures) contains *rsa*",
                note="no RSA figure found",
            )
        )
        notes = (notes + "; " if notes else "") + "no rsa figures found"
    else:
        notes = (notes + "; " if notes and notes != "ok" else notes) + f"rsa_figs_found={len(rsa_figs)}"

    add_item(
        item_id="L3-3",
        title="RSA summary",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes=notes or "ok",
        extra={"details": evidence, "rsa_figs": rsa_figs[:50]},
    )

    # 4. CCD summary (hardneg)
    item_id = "L2-1-4"
    base = "cache/model_eval_results/shared982_ccd"
    req_files = [
        f"{base}/ccd_summary.csv",
        f"{base}/ccd_summary.md",
        f"{base}/figures/figures_manifest.json",
        f"{base}/figures/figures_report.md",
    ]
    paths = []
    statuses = []
    for rel in req_files:
        st, p = _check_file(root, rel, min_size=10, missing=missing, item_id=item_id, severity="CRITICAL")
        statuses.append(st)
        paths.append(p)

    # Fig01-Fig08 size check
    fig_dir_rel = f"{base}/figures"
    fig_dir = root / fig_dir_rel
    fig_missing: List[str] = []
    fig_small: List[str] = []
    for i in range(1, 9):
        pat = f"Fig{i:02d}"
        matches = [p for p in fig_dir.glob(f"*{pat}*") if p.is_file()]
        if not matches:
            fig_missing.append(f"{fig_dir_rel}/{pat}*")
            missing.append(MissingRow(item_id=item_id, severity="CRITICAL", expected_path=f"{fig_dir_rel}/{pat}*", note="missing Fig01-Fig08"))
            continue
        for m in matches:
            st = _safe_stat(m)
            if st.get("size_bytes", 0) < 5 * 1024:
                fig_small.append(str(m.relative_to(root)))
                missing.append(
                    MissingRow(
                        item_id=item_id,
                        severity="CRITICAL",
                        expected_path=str(m.relative_to(root)),
                        note="figure file too small (<5KB)",
                    )
                )

    if fig_missing or fig_small:
        statuses.append(STATUS_FAIL)

    status = _status_merge(statuses)
    notes = "ok"
    if fig_missing:
        notes = f"missing: {fig_missing}"
    if fig_small:
        notes = (notes + "; " if notes != "ok" else "") + f"too_small: {fig_small}"

    add_item(
        item_id=item_id,
        title="CCD summary（hardneg）",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes=notes,
        extra={"fig_missing": fig_missing, "fig_small": fig_small},
    )

    # 5. CCD by-type
    item_id = "L2-1-5"
    rel = f"{base}/ccd_by_type.csv"
    paths = []
    st, p = _check_file(root, rel, min_size=10, missing=missing, item_id=item_id, severity="CRITICAL")
    paths.append(p)
    status = st
    notes = p.get("notes", "")
    details: Dict[str, Any] = {}
    if st != STATUS_FAIL:
        try:
            rows = _read_csv_rows(root / rel)
            types = {r.get("type", "").strip() for r in rows if r.get("type", "").strip()}
            required = {"object", "attribute", "relation"}
            missing_types = sorted(required - {t.lower() for t in types})
            details = {"types_found": sorted({t.lower() for t in types}), "row_count": len(rows)}
            if missing_types:
                status = STATUS_FAIL
                notes = f"missing types: {missing_types}"
                missing.append(
                    MissingRow(
                        item_id=item_id,
                        severity="CRITICAL",
                        expected_path=rel,
                        note=f"missing required types {missing_types}",
                    )
                )
            else:
                notes = "ok"
        except Exception as e:
            status = STATUS_FAIL
            notes = f"failed to parse csv: {e}"

    add_item(
        item_id=item_id,
        title="CCD by-type",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes=notes,
        extra={"details": details},
    )

    # 6. Negative quality audit
    item_id = "L2-1-6"
    audit_dir_rel = f"{base}/audit"
    audit_dir = root / audit_dir_rel
    req = [
        f"{audit_dir_rel}/audit_tables.csv",
        f"{audit_dir_rel}/Fig_audit_len_chars.png",
        f"{audit_dir_rel}/Fig_audit_len_words.png",
        f"{audit_dir_rel}/Fig_audit_negation_rate.png",
        f"{audit_dir_rel}/Fig_audit_sim_text.png",
        f"{audit_dir_rel}/Fig_audit_type_coverage.png",
    ]
    paths = []
    statuses = []

    if not audit_dir.exists():
        statuses.append(STATUS_FAIL)
        missing.append(MissingRow(item_id=item_id, severity="CRITICAL", expected_path=audit_dir_rel + "/", note="missing audit directory"))
        paths.append({"exists": False, "path": str(audit_dir)})
    else:
        paths.append(_safe_stat(audit_dir))

    for rel in req:
        min_size = 10
        if rel.lower().endswith(".png"):
            min_size = 1024
        st, p = _check_file(root, rel, min_size=min_size, missing=missing, item_id=item_id, severity="CRITICAL")
        statuses.append(st)
        paths.append(p)

    status = _status_merge(statuses) if statuses else STATUS_PASS
    add_item(
        item_id=item_id,
        title="Negative quality audit",
        severity="CRITICAL",
        status=status,
        paths=paths,
        notes="ok" if status == STATUS_PASS else "missing or invalid files",
    )

    # 7. IS-RSA matrices + delta figs
    item_id = "B-7"
    base_isrsa = "cache/model_eval_results/shared982_isrsa"
    req = [
        f"{base_isrsa}/baseline/isrsa_matrix.csv",
        f"{base_isrsa}/textalign_llm/isrsa_matrix.csv",
        "results/tables/isrsa_summary.csv",
        f"{base_isrsa}/figures/Fig_isrsa_heatmap_baseline.png",
        f"{base_isrsa}/figures/Fig_isrsa_heatmap_textalign_llm.png",
        f"{base_isrsa}/figures/Fig_isrsa_delta.png",
        f"{base_isrsa}/figures/Fig_mean_cos_delta.png",
        f"{base_isrsa}/figures/figures_manifest.json",
        f"{base_isrsa}/figures/figures_report.md",
    ]
    paths = []
    statuses = []
    for rel in req:
        min_size = 10
        if rel.lower().endswith(".png"):
            min_size = 5 * 1024
        st, p = _check_file(root, rel, min_size=min_size, missing=missing, item_id=item_id, severity="MAJOR")
        statuses.append(st)
        paths.append(p)

    add_item(
        item_id=item_id,
        title="IS-RSA matrices + delta figs",
        severity="MAJOR",
        status=_status_merge(statuses),
        paths=paths,
        notes="ok" if _status_merge(statuses) == STATUS_PASS else "missing or invalid files",
    )

    # 8. Ablation-K
    item_id = "C-8"
    rel = "results/tables/ccd_ablation_k.csv"
    paths = []
    st, p = _check_file(root, rel, min_size=10, missing=missing, item_id=item_id, severity="MAJOR")
    paths.append(p)
    status = st
    notes = p.get("notes", "")
    details = {}
    if st != STATUS_FAIL:
        try:
            rows = _read_csv_rows(root / rel)
            k_values: set[int] = set()
            unavailable_ok = False
            for r in rows:
                hv = r.get("hardneg_k", r.get("k_neg", r.get("k", "")))
                try:
                    k = int(float(hv))
                    if k > 0:
                        k_values.add(k)
                except Exception:
                    pass
                note = (r.get("note", "") or "").lower()
                if "unavailable" in note:
                    unavailable_ok = True

            missing_k = [k for k in (2, 4) if k not in k_values]
            if missing_k:
                status = STATUS_FAIL
                notes = f"missing required K rows: {missing_k}"
                missing.append(MissingRow(item_id=item_id, severity="MAJOR", expected_path=rel, note=notes))
            else:
                # If K=8 present but no unavailable note, warn.
                if 8 in k_values and not unavailable_ok:
                    status = STATUS_WARN
                    notes = "K=8 present but no 'unavailable' note detected"
                else:
                    notes = "ok"
            details = {"k_values_found": sorted(k_values), "k8_unavailable_note_found": unavailable_ok}
        except Exception as e:
            status = STATUS_FAIL
            notes = f"failed to parse csv: {e}"

    add_item(
        item_id=item_id,
        title="Ablation-K",
        severity="MAJOR",
        status=status,
        paths=paths,
        notes=notes,
        extra={"details": details},
    )

    # 9. Ablation-difficulty (hardest vs random, K_fixed=2)
    item_id = "C-9"
    rel_table = "results/tables/ccd_ablation_difficulty.csv"
    rel_fig = "cache/model_eval_results/shared982_ccd/figures/Fig09_ccd_ablation_difficulty.png"
    rel_manifest = "cache/model_eval_results/shared982_ccd/figures/figures_manifest.json"
    rel_report = "cache/model_eval_results/shared982_ccd/figures/figures_report.md"

    paths = []
    statuses = []
    for rel in [rel_table, rel_fig]:
        min_size = 10
        if rel.lower().endswith(".png"):
            min_size = 5 * 1024
        st, p = _check_file(root, rel, min_size=min_size, missing=missing, item_id=item_id, severity="MAJOR")
        statuses.append(st)
        paths.append(p)

    fig09_in_manifest = False
    fig09_in_report = False
    if (root / rel_manifest).exists():
        try:
            txt = _read_text(root / rel_manifest)
            fig09_in_manifest = "Fig09" in txt
        except Exception:
            fig09_in_manifest = False
    if (root / rel_report).exists():
        try:
            txt = _read_text(root / rel_report)
            fig09_in_report = "Fig09" in txt
        except Exception:
            fig09_in_report = False

    if not fig09_in_manifest or not fig09_in_report:
        statuses.append(STATUS_FAIL)
        missing.append(
            MissingRow(
                item_id=item_id,
                severity="MAJOR",
                expected_path=rel_manifest + " and " + rel_report,
                note="Fig09 entry missing from manifest/report",
            )
        )

    add_item(
        item_id=item_id,
        title="Ablation-difficulty（hardest vs random, K_fixed=2）",
        severity="MAJOR",
        status=_status_merge(statuses),
        paths=paths,
        notes="ok" if _status_merge(statuses) == STATUS_PASS else "missing or invalid files",
        extra={"details": {"fig09_in_manifest": fig09_in_manifest, "fig09_in_report": fig09_in_report}},
    )

    # 10. Efficiency summary + figs
    item_id = "A-10"
    rel_table = "results/tables/efficiency_summary.csv"
    rel_fig1 = "results/figures_main/Fig_efficiency_ccd_acc1.png"
    rel_fig2 = "results/figures_main/Fig_efficiency_twoafc_hard.png"
    rel_manifest = "cache/model_eval_results/shared982_efficiency/figures/figures_manifest.json"
    rel_report = "cache/model_eval_results/shared982_efficiency/figures/figures_report.md"

    req = [rel_table, rel_fig1, rel_fig2, rel_manifest, rel_report]
    paths = []
    statuses = []

    for rel in req:
        min_size = 10
        if rel.lower().endswith(".png"):
            min_size = 5 * 1024
        st, p = _check_file(root, rel, min_size=min_size, missing=missing, item_id=item_id, severity="CRITICAL")
        statuses.append(st)
        paths.append(p)

    details = {}
    if (root / rel_table).exists():
        try:
            n_data = _csv_nonempty_data_lines(root / rel_table)
            details["data_row_count_excluding_header"] = n_data
            if n_data != 12:
                statuses.append(STATUS_FAIL)
                missing.append(
                    MissingRow(
                        item_id=item_id,
                        severity="CRITICAL",
                        expected_path=rel_table,
                        note=f"expected 12 data rows (excluding header), got {n_data}",
                    )
                )
        except Exception as e:
            statuses.append(STATUS_FAIL)
            missing.append(MissingRow(item_id=item_id, severity="CRITICAL", expected_path=rel_table, note=f"failed to count rows: {e}"))

    # cache completeness: embeds/**/brain.npy & ids.npy for 12 combos
    embeds_root_rel = "cache/model_eval_results/shared982_efficiency/embeds"
    embeds_root = root / embeds_root_rel
    if not embeds_root.exists():
        statuses.append(STATUS_FAIL)
        missing.append(
            MissingRow(
                item_id=item_id,
                severity="CRITICAL",
                expected_path=embeds_root_rel + "/",
                note="missing embeds directory (expected embeds/**/brain.npy and ids.npy for 12 combos)",
            )
        )
        details["embeds_dir_exists"] = False
    else:
        details["embeds_dir_exists"] = True

    # metrics per 12 combos from efficiency_summary.csv
    expected_metric_missing: List[str] = []
    metric_bootstrap_bad: List[str] = []

    if (root / rel_table).exists():
        try:
            rows = _read_csv_rows(root / rel_table)
            # Normalize setting field to folder names used under metrics/
            def setting_to_folder(setting: str) -> str:
                s = setting.strip().lower()
                if s.endswith("sess"):
                    return s
                # allow numeric settings like 1/2/40
                if s.isdigit():
                    return f"{s}sess"
                return s

            metric_paths: List[Path] = []
            for r in rows:
                subj = str(r.get("subj", "")).strip()
                model = str(r.get("model", "")).strip()
                setting = setting_to_folder(str(r.get("setting", "")).strip())
                seed = str(r.get("seed", "0")).strip() or "0"
                if not subj or not model or not setting:
                    continue
                subj_folder = f"subj{subj.zfill(2)}" if len(subj) <= 2 else f"subj{subj}"
                base_metrics = root / "cache/model_eval_results/shared982_efficiency/metrics" / subj_folder / model / setting / f"seed{seed}"
                for fn in ["ccd.json", "rsa.json", "twoafc_hard.json"]:
                    metric_paths.append(base_metrics / fn)

            # Parallel read jsons for speed.
            def check_metric(p: Path) -> Tuple[Path, str, Optional[int]]:
                if not p.exists():
                    return p, "missing", None
                try:
                    obj = _load_json(p)
                    bs = None
                    if isinstance(obj, dict) and "bootstrap" in obj:
                        try:
                            bs = int(obj["bootstrap"])
                        except Exception:
                            bs = None
                    return p, "ok", bs
                except Exception:
                    return p, "bad_json", None

            with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8))) as ex:
                futs = [ex.submit(check_metric, p) for p in metric_paths]
                for fut in as_completed(futs):
                    p, state, bs = fut.result()
                    relp = str(p.relative_to(root))
                    if state != "ok":
                        expected_metric_missing.append(relp)
                    else:
                        if bs is not None and bs != 1000:
                            metric_bootstrap_bad.append(f"{relp} bootstrap={bs}")

            if expected_metric_missing:
                statuses.append(STATUS_FAIL)
                for relp in expected_metric_missing[:200]:
                    missing.append(MissingRow(item_id=item_id, severity="CRITICAL", expected_path=relp, note="missing efficiency metric json"))

            if metric_bootstrap_bad:
                statuses.append(STATUS_WARN)

            details["expected_metric_json_count"] = len(metric_paths)
            details["missing_metric_json_count"] = len(expected_metric_missing)
            details["bootstrap_mismatch_count"] = len(metric_bootstrap_bad)
        except Exception as e:
            statuses.append(STATUS_FAIL)
            missing.append(MissingRow(item_id=item_id, severity="CRITICAL", expected_path=rel_table, note=f"failed to validate metrics: {e}"))

    if metric_bootstrap_bad:
        details["bootstrap_mismatches"] = metric_bootstrap_bad[:50]

    add_item(
        item_id=item_id,
        title="Efficiency summary + figs",
        severity="CRITICAL",
        status=_status_merge(statuses),
        paths=paths,
        notes="ok" if _status_merge(statuses) == STATUS_PASS else "missing or invalid files",
        extra={"details": details},
    )

    # --- Protocol consistency checks ---
    protocol_checks: List[Dict[str, Any]] = []

    # (1) shared982 N consistency
    n_obs: List[Tuple[str, Optional[int]]] = []

    def obs_n(label: str, v: Optional[int]) -> None:
        n_obs.append((label, v))

    # From main_results.csv columns
    if main_rows:
        # Use first valid row per metric
        for col in ["retrieval_N", "ccd_N", "isrsa_N"]:
            vs = []
            for r in main_rows:
                fv = _parse_float_maybe(r.get(col, ""))
                if fv is not None and fv > 0:
                    vs.append(int(round(fv)))
            if vs:
                obs_n(f"results/tables/main_results.csv:{col}", sorted(set(vs))[0])

    # From isrsa metrics
    for rel in [f"{base_isrsa}/baseline/metrics.json", f"{base_isrsa}/textalign_llm/metrics.json"]:
        p = root / rel
        if p.exists():
            try:
                obj = _load_json(p)
                if isinstance(obj, dict) and "N" in obj:
                    obs_n(rel + ":N", int(obj["N"]))
            except Exception:
                pass

    # From isrsa_summary
    isrsa_summary = root / "results/tables/isrsa_summary.csv"
    if isrsa_summary.exists():
        try:
            rows = _read_csv_rows(isrsa_summary)
            ns = {int(float(r.get("N", "0") or 0)) for r in rows if (r.get("N", "") or "").strip()}
            if ns:
                obs_n("results/tables/isrsa_summary.csv:N", sorted(ns)[0])
        except Exception:
            pass

    # From efficiency_summary
    eff_summary = root / "results/tables/efficiency_summary.csv"
    if eff_summary.exists():
        try:
            rows = _read_csv_rows(eff_summary)
            ns = {int(float(r.get("N", "0") or 0)) for r in rows if (r.get("N", "") or "").strip()}
            if ns:
                obs_n("results/tables/efficiency_summary.csv:N", sorted(ns)[0])
        except Exception:
            pass

    # From CCD summary
    ccd_summary = root / f"{base}/ccd_summary.csv"
    if ccd_summary.exists():
        try:
            rows = _read_csv_rows(ccd_summary, max_rows=2000)
            ns = {int(float(r.get("n_eval", "0") or 0)) for r in rows if (r.get("n_eval", "") or "").strip()}
            if ns:
                obs_n(f"{base}/ccd_summary.csv:n_eval", sorted(ns)[0])
        except Exception:
            pass

    expected_n = 982
    # Allow a known exception: CCD hardneg uses a valid-mask subset (909) by design.
    allowed_ccd_n = 909

    n_bad_raw = [(lbl, v) for (lbl, v) in n_obs if v is not None and v != expected_n]
    n_bad: List[Tuple[str, int]] = []
    n_allowed: List[Tuple[str, int]] = []
    for lbl, v in n_bad_raw:
        if v == allowed_ccd_n and re.search(r"ccd", lbl, re.IGNORECASE):
            n_allowed.append((lbl, v))
        else:
            n_bad.append((lbl, v))

    n_status = STATUS_PASS if not n_bad else STATUS_WARN
    if n_bad:
        n_note = "found N values not equal to 982"
    elif n_allowed:
        n_note = "ok (CCD hardneg uses a valid hardneg subset; expected 909 for CCD hardneg)"
    else:
        n_note = "ok"

    protocol_checks.append(
        {
            "check": "shared982 N consistency",
            "status": n_status,
            "expected": expected_n,
            "observed": [{"source": lbl, "value": v} for (lbl, v) in n_obs],
            "allowed_exceptions": {"ccd_hardneg_n": allowed_ccd_n} if n_allowed else {},
            "notes": n_note,
        }
    )

    # (2) bootstrap consistency
    bs_obs: List[Tuple[str, Optional[int]]] = []

    def obs_bs(label: str, v: Optional[int]) -> None:
        bs_obs.append((label, v))

    # CCD summary bootstrap column
    if ccd_summary.exists():
        try:
            rows = _read_csv_rows(ccd_summary, max_rows=5000)
            bss = {int(float(r.get("bootstrap", "0") or 0)) for r in rows if (r.get("bootstrap", "") or "").strip()}
            for v in sorted(bss):
                obs_bs(f"{base}/ccd_summary.csv:bootstrap", v)
        except Exception:
            pass

    # ISRSA metrics
    for rel in [f"{base_isrsa}/baseline/metrics.json", f"{base_isrsa}/textalign_llm/metrics.json"]:
        p = root / rel
        if p.exists():
            try:
                obj = _load_json(p)
                if isinstance(obj, dict) and "bootstrap" in obj:
                    obs_bs(rel + ":bootstrap", int(obj["bootstrap"]))
            except Exception:
                pass

    # Efficiency metric jsons (derive from efficiency_summary rows)
    eff_metric_bootstrap_bad: List[str] = []
    eff_metric_bootstrap_obs: List[Tuple[str, Optional[int]]] = []
    if eff_summary.exists():
        try:
            rows = _read_csv_rows(eff_summary)
            metric_paths: List[Path] = []
            for r in rows:
                subj = str(r.get("subj", "")).strip()
                model = str(r.get("model", "")).strip()
                setting = str(r.get("setting", "")).strip().lower()
                seed = str(r.get("seed", "0")).strip() or "0"
                if not subj or not model or not setting:
                    continue
                if setting.isdigit():
                    setting = f"{setting}sess"
                subj_folder = f"subj{subj.zfill(2)}" if len(subj) <= 2 else f"subj{subj}"
                base_metrics = root / "cache/model_eval_results/shared982_efficiency/metrics" / subj_folder / model / setting / f"seed{seed}"
                for fn in ["ccd.json", "rsa.json", "twoafc_hard.json"]:
                    metric_paths.append(base_metrics / fn)

            def read_bs(p: Path) -> Tuple[str, Optional[int]]:
                relp = str(p.relative_to(root))
                if not p.exists():
                    return relp, None
                try:
                    obj = _load_json(p)
                    if isinstance(obj, dict) and "bootstrap" in obj:
                        return relp, int(obj["bootstrap"])
                    return relp, None
                except Exception:
                    return relp, None

            with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8))) as ex:
                futs = [ex.submit(read_bs, p) for p in metric_paths]
                for fut in as_completed(futs):
                    relp, v = fut.result()
                    eff_metric_bootstrap_obs.append((relp, v))
                    if v is not None and v != 1000:
                        eff_metric_bootstrap_bad.append(f"{relp} bootstrap={v}")
        except Exception:
            pass

    for relp, v in eff_metric_bootstrap_obs[:50]:
        if v is not None:
            obs_bs(relp + ":bootstrap", v)

    bs_expected = 1000
    bs_bad = [(lbl, v) for (lbl, v) in bs_obs if v is not None and v != bs_expected]
    bs_status = STATUS_PASS if not bs_bad and not eff_metric_bootstrap_bad else STATUS_WARN
    protocol_checks.append(
        {
            "check": "bootstrap consistency",
            "status": bs_status,
            "expected": bs_expected,
            "observed_examples": [{"source": lbl, "value": v} for (lbl, v) in bs_obs[:20]],
            "mismatches": eff_metric_bootstrap_bad[:50],
            "notes": "ok" if bs_status == STATUS_PASS else "bootstrap mismatch detected in some metrics",
        }
    )

    # (3) CLIP dimension consistency (clip_img_gt.npy vs brain.npy => 1664)
    # clip_img_gt.npy may not exist in this repo; brain.npy may be absent.
    clip_candidates = list(root.rglob("clip_img_gt.npy"))
    brain_candidates = list(root.rglob("brain.npy"))

    clip_dim = None
    clip_msg = "clip_img_gt.npy not found"
    if clip_candidates:
        clip_dim, clip_msg = _npy_feature_dim(clip_candidates[0])

    brain_dim = None
    brain_msg = "brain.npy not found"
    if brain_candidates:
        brain_dim, brain_msg = _npy_feature_dim(brain_candidates[0])

    dim_expected = 1664
    dim_status = STATUS_PASS
    dim_notes = []

    if clip_candidates and clip_dim is not None and clip_dim != dim_expected:
        dim_status = STATUS_WARN
        dim_notes.append(f"clip_img_gt.npy dim={clip_dim} (expected {dim_expected})")
    if brain_candidates and brain_dim is not None and brain_dim != dim_expected:
        dim_status = STATUS_WARN
        dim_notes.append(f"brain.npy dim={brain_dim} (expected {dim_expected})")

    if not clip_candidates or not brain_candidates:
        # Can't fully validate; warn.
        dim_status = STATUS_WARN
        if not clip_candidates:
            dim_notes.append("clip_img_gt.npy missing")
        if not brain_candidates:
            dim_notes.append("brain.npy missing")

    # Add additional evidence from ISRSA metrics.json D field
    isrsa_d = None
    try:
        p = root / f"{base_isrsa}/baseline/metrics.json"
        if p.exists():
            obj = _load_json(p)
            if isinstance(obj, dict) and "D" in obj:
                isrsa_d = int(obj["D"])
    except Exception:
        isrsa_d = None

    protocol_checks.append(
        {
            "check": "CLIP feature dim consistency",
            "status": dim_status,
            "expected": dim_expected,
            "clip_img_gt": {
                "found": bool(clip_candidates),
                "path": str(clip_candidates[0].relative_to(root)) if clip_candidates else None,
                "dim": clip_dim,
                "notes": clip_msg,
            },
            "brain": {
                "found": bool(brain_candidates),
                "path": str(brain_candidates[0].relative_to(root)) if brain_candidates else None,
                "dim": brain_dim,
                "notes": brain_msg,
            },
            "extra_evidence": {"shared982_isrsa_baseline.metrics.json:D": isrsa_d},
            "notes": "; ".join(dim_notes) if dim_notes else "ok",
        }
    )

    # (4) main_results.csv duplicates by group+subj+tag
    dup_status = STATUS_PASS
    dup_details = {}
    if main_rows:
        dup_status, dup_details = _audit_duplicates_main_results(main_rows)
    else:
        dup_status = STATUS_FAIL
        dup_details = {"notes": "main_results.csv missing or unreadable"}

    protocol_checks.append(
        {
            "check": "main_results.csv duplicate rows (group+subj+tag)",
            "status": dup_status,
            "details": dup_details,
        }
    )

    # Produce missing csv rows (sorted)
    missing_sorted = sorted(missing, key=lambda r: (r.severity, r.item_id, r.expected_path))

    # Summary counts
    pass_n = sum(1 for it in items_out if it["status"] == STATUS_PASS)
    warn_n = sum(1 for it in items_out if it["status"] == STATUS_WARN)
    fail_n = sum(1 for it in items_out if it["status"] == STATUS_FAIL)

    missing_sev: Dict[str, int] = {}
    for m in missing_sorted:
        missing_sev[m.severity] = missing_sev.get(m.severity, 0) + 1

    # Render MD
    md_lines: List[str] = []
    md_lines.append("# Artifact audit report")
    md_lines.append("")
    md_lines.append(f"- generated_utc: {_utc_iso(_dt.datetime.utcnow().timestamp())}")
    md_lines.append(f"- root: {root}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- PASS: {pass_n}")
    md_lines.append(f"- WARN: {warn_n}")
    md_lines.append(f"- FAIL: {fail_n}")
    md_lines.append(f"- missing_entries: {len(missing_sorted)}")
    if missing_sev:
        md_lines.append(f"- missing_by_severity: {json.dumps(missing_sev, ensure_ascii=False)}")
    md_lines.append("")

    md_lines.append("## Checklist")
    md_lines.append("")
    md_lines.append("| item_id | severity | status | title |")
    md_lines.append("|---|---|---|---|")
    for it in items_out:
        md_lines.append(f"| {it['item_id']} | {it['severity']} | {it['status']} | {it['title']} |")

    md_lines.append("")
    md_lines.append("## Details")
    md_lines.append("")
    for it in items_out:
        md_lines.append(f"### {it['item_id']}: {it['title']}")
        md_lines.append("")
        md_lines.append(f"- severity: {it['severity']}")
        md_lines.append(f"- status: {it['status']}")
        md_lines.append(f"- notes: {it.get('notes','')}")
        md_lines.append("")
        md_lines.append("Paths:")
        for p in it.get("paths", []):
            if isinstance(p, dict):
                relp = None
                try:
                    relp = str(Path(p.get("path", "")).resolve().relative_to(root))
                except Exception:
                    relp = p.get("path", "")
                md_lines.append(
                    f"- {relp} | exists={p.get('exists')} size={p.get('size_bytes')} mtime={p.get('mtime_iso', p.get('mtime'))} | {p.get('notes','')}"
                )
        if "details" in it:
            md_lines.append("")
            md_lines.append("Details:")
            md_lines.append("```json")
            md_lines.append(json.dumps(it["details"], ensure_ascii=False, indent=2)[:40000])
            md_lines.append("```")
        if "rsa_figs" in it:
            md_lines.append("")
            md_lines.append(f"RSA figs (sample): {it['rsa_figs'][:10]}")
        md_lines.append("")

    md_lines.append("## Protocol consistency checks")
    md_lines.append("")
    for chk in protocol_checks:
        md_lines.append(f"### {chk['check']}")
        md_lines.append("")
        md_lines.append(f"- status: {chk['status']}")
        notes = chk.get("notes")
        if notes:
            md_lines.append(f"- notes: {notes}")
        md_lines.append("```json")
        md_lines.append(json.dumps(chk, ensure_ascii=False, indent=2)[:40000])
        md_lines.append("```")
        md_lines.append("")

    md_lines.append("## Missing items")
    md_lines.append("")
    if not missing_sorted:
        md_lines.append("(none)")
    else:
        md_lines.append("| item_id | severity | expected_path | note |")
        md_lines.append("|---|---|---|---|")
        for m in missing_sorted:
            md_lines.append(f"| {m.item_id} | {m.severity} | {m.expected_path} | {m.note} |")

    audit_md = out_md if out_md is not None else (results_dir / "ARTIFACT_AUDIT.md")
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    audit_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    audit_json = out_json if out_json is not None else (results_dir / "ARTIFACT_AUDIT.json")
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(
        json.dumps(
            {
                "generated_utc": _utc_iso(_dt.datetime.utcnow().timestamp()),
                "root": str(root),
                "summary": {
                    "PASS": pass_n,
                    "WARN": warn_n,
                    "FAIL": fail_n,
                    "missing_entries": len(missing_sorted),
                    "missing_by_severity": missing_sev,
                },
                "items": items_out,
                "protocol_consistency_checks": protocol_checks,
                "missing": [m.__dict__ for m in missing_sorted],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    missing_csv = out_missing_csv if out_missing_csv is not None else (results_dir / "ARTIFACT_MISSING.csv")
    missing_csv.parent.mkdir(parents=True, exist_ok=True)
    if missing_sorted:
        with missing_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "severity", "expected_path", "note"])
            for m in missing_sorted:
                w.writerow([m.item_id, m.severity, m.expected_path, m.note])
    else:
        if missing_csv.exists():
            try:
                missing_csv.unlink()
            except Exception:
                pass

    return {
        "summary": {"PASS": pass_n, "WARN": warn_n, "FAIL": fail_n, "missing_entries": len(missing_sorted), "missing_by_severity": missing_sev},
        "audit_md": str(audit_md),
        "audit_json": str(audit_json),
        "missing_csv": str(missing_csv),
        "missing_count": len(missing_sorted),
    }


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit result artifacts (read-only)")
    ap.add_argument("--root", type=str, default=None, help="repo root (default: auto-detect from this script location)")
    ap.add_argument("--out-md", type=str, default=None, help="output markdown path (relative to --root unless absolute)")
    ap.add_argument("--out-json", type=str, default=None, help="output json path (relative to --root unless absolute)")
    ap.add_argument(
        "--out-missing-csv",
        type=str,
        default=None,
        help="output missing.csv path (relative to --root unless absolute)",
    )
    args = ap.parse_args(argv)

    if args.root is None:
        root = Path(__file__).resolve().parents[1]
    else:
        root = Path(args.root).expanduser().resolve()

    def _resolve_out(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        pp = Path(p).expanduser()
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    out = run_audit(
        root,
        out_md=_resolve_out(args.out_md),
        out_json=_resolve_out(args.out_json),
        out_missing_csv=_resolve_out(args.out_missing_csv),
    )
    s = out["summary"]
    sev = s.get("missing_by_severity", {})
    sev_str = ", ".join([f"{k}={v}" for k, v in sev.items()]) if sev else "none"

    print("[ARTIFACT AUDIT SUMMARY]")
    print(f"PASS={s['PASS']} WARN={s['WARN']} FAIL={s['FAIL']} missing_entries={s['missing_entries']}")
    print(f"missing_by_severity: {sev_str}")
    print(f"wrote: {out['audit_md']}")
    print(f"wrote: {out['audit_json']}")
    if out["missing_count"]:
        print(f"wrote: {out['missing_csv']}")

    # Exit non-zero if any FAIL.
    return 0 if s["FAIL"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
