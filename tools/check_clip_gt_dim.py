#!/usr/bin/env python3
"""Check CLIP GT embedding dimension (clip_img_gt.npy).

This is a read-only verification utility (no training/inference/eval).

It writes a small human-readable report to:
  results/audit/clip_gt_check.txt

Usage:
  python tools/check_clip_gt_dim.py
  python tools/check_clip_gt_dim.py --root /path/to/repo
  python tools/check_clip_gt_dim.py --clip-path /some/path/clip_img_gt.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _infer_dim(shape: Tuple[int, ...]) -> Optional[int]:
    if not shape:
        return None
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[-1])


def _find_clip_candidates(root: Path) -> List[Path]:
    return sorted(root.rglob("clip_img_gt.npy"))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Check clip_img_gt.npy dim")
    ap.add_argument("--root", type=str, default=None, help="repo root (default: auto-detect from this script location)")
    ap.add_argument("--clip-path", type=str, default=None, help="explicit path to clip_img_gt.npy")
    ap.add_argument(
        "--out",
        type=str,
        default="results/audit/clip_gt_check.txt",
        help="output report path (relative to --root unless absolute)",
    )
    ap.add_argument("--expected-dim", type=int, default=1664)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1] if args.root is None else Path(args.root).expanduser().resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates: List[Path] = []
    if args.clip_path:
        candidates.append(Path(args.clip_path).expanduser().resolve())
    else:
        candidates.extend(_find_clip_candidates(root))
        # Also check the known external default, if present.
        ext = Path("/mnt/work/data_cache/clip_img_gt.npy")
        if ext.exists():
            candidates.append(ext)

    # De-dup while preserving order.
    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        uniq.append(p)
    candidates = uniq

    np = None
    try:
        import numpy as _np  # type: ignore

        np = _np
    except Exception as e:
        msg = f"numpy import failed: {type(e).__name__}: {e}"
        out_path.write_text(msg + "\n", encoding="utf-8")
        print(msg, file=sys.stderr)
        return 2

    lines: List[str] = []
    lines.append(f"root: {root}")
    lines.append(f"expected_dim: {args.expected_dim}")
    lines.append("")

    ok_any = False
    for p in candidates:
        lines.append(f"path: {p}")
        if not p.exists():
            lines.append("  exists: false")
            lines.append("")
            continue
        try:
            arr = np.load(str(p), mmap_mode="r")
            shape = tuple(getattr(arr, "shape", ()) or ())
            dim = _infer_dim(shape)
            lines.append("  exists: true")
            lines.append(f"  shape: {shape}")
            lines.append(f"  dtype: {getattr(arr, 'dtype', None)}")
            lines.append(f"  inferred_dim: {dim}")
            lines.append(f"  dim_ok: {dim == args.expected_dim}")
            if dim == args.expected_dim:
                ok_any = True
        except Exception as e:
            lines.append(f"  load_error: {type(e).__name__}: {e}")
        lines.append("")

    lines.append(f"overall_ok_any: {ok_any}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    return 0 if ok_any else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
