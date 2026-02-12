#!/usr/bin/env python
# coding: utf-8
"""Wrapper to run CCD-Hard with hardneg mask filtering.

Uses rerun_all_ccd_shared982.py under the hood but filters hardneg JSONL
to only include samples in the valid mask.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]


def filter_hardneg_jsonl(hardneg_jsonl_path: Path, valid_mask_path: Path) -> Path:
    """Create a *single-neg-per-image* hardneg JSONL for the valid shared982 subset.

    - Input `shared982_hardneg.jsonl` typically contains multiple candidates per image (by type).
    - `eval_ccd_embed.py` expects exactly one `neg_caption` per `image_id` (it overwrites duplicates).
    - We pick the row with max `sim_text` for each image.

    Returns: path to filtered JSONL (cached under cache/hardneg/).
    """
    import json

    valid_mask = np.load(valid_mask_path)
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 1:
        raise RuntimeError(f"valid_mask must be 1D bool array, got shape={valid_mask.shape} dtype={valid_mask.dtype}")

    m982 = np.load(PROJ / "src" / "shared982.npy")
    ids982 = np.where(m982)[0].astype(np.int64) if m982.dtype == np.bool_ else m982.astype(np.int64)
    if int(valid_mask.shape[0]) != int(ids982.shape[0]):
        raise RuntimeError(
            f"valid_mask length={int(valid_mask.shape[0])} != shared982 size={int(ids982.shape[0])}"
        )

    valid_coco_ids = set(ids982[valid_mask].tolist())
    print(f"[Filter] shared982 valid images: {len(valid_coco_ids)}/{int(ids982.shape[0])}")

    best = {}  # image_id -> (sim, json_str)
    with open(hardneg_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "image_id" not in row or "neg_caption" not in row:
                continue
            image_id = int(row["image_id"])
            if image_id not in valid_coco_ids:
                continue
            sim = row.get("sim_text", None)
            try:
                sim_val = float(sim) if sim is not None else float("-inf")
            except Exception:
                sim_val = float("-inf")
            cur = best.get(image_id)
            if cur is None or sim_val > cur[0]:
                # store original row (preserve extra fields like type)
                best[image_id] = (sim_val, json.dumps(row, ensure_ascii=False) + "\n")

    missing = [int(i) for i in ids982[valid_mask] if int(i) not in best]
    if missing:
        raise RuntimeError(f"Hardneg missing for {len(missing)} valid images; cannot run CCD-Hard. Example: {missing[:5]}")

    # Write one line per image, in shared982 canonical order
    n_valid = int(valid_mask.sum())
    filtered_path = PROJ / "cache" / "hardneg" / f"shared982_hardneg_single_{n_valid}.jsonl"
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, "w", encoding="utf-8") as out:
        for image_id in ids982[valid_mask].tolist():
            out.write(best[int(image_id)][1])

    print(f"[Filter] Wrote {len(best)} lines (1/image) to {filtered_path}")
    return filtered_path


def main():
    # Get paths from env or use defaults
    hardneg_jsonl = Path(os.environ.get("CCD_HARD_NEG_JSONL", str(PROJ / "cache" / "hardneg" / "shared982_hardneg.jsonl")))
    valid_mask = Path(os.environ.get("CCD_HARD_NEG_MASK", str(PROJ / "cache" / "hardneg" / "hardneg_valid_mask.npy")))
    
    assert hardneg_jsonl.is_file(), f"Not found: {hardneg_jsonl}"
    assert valid_mask.is_file(), f"Not found: {valid_mask}"
    
    print(f"[Step 2] CCD-Hard Batch Run with Hardneg Mask Filter")
    print(f"  hardneg JSONL: {hardneg_jsonl}")
    print(f"  valid mask: {valid_mask}")
    
    # Filter hardneg JSONL
    filtered_jsonl = filter_hardneg_jsonl(hardneg_jsonl, valid_mask)

    if os.environ.get("CCD_FILTER_ONLY", "").strip() in ("1", "true", "yes"):
        print("[Step 2] CCD_FILTER_ONLY is set; skipping batch run.")
        return
    
    # Set env and run rerun_all_ccd_shared982.py
    env = os.environ.copy()
    env.update({
        "CCD_HARD_NEG_JSONL": str(filtered_jsonl),
        "CCD_EVAL_KEEP_MASK_NPY": str(valid_mask),
        "CCD_BOOTSTRAP": os.environ.get("CCD_BOOTSTRAP", "1000"),
        "CCD_K_NEG": os.environ.get("CCD_K_NEG", "31"),
        "CCD_SEED": os.environ.get("CCD_SEED", "42"),
    })
    
    print(f"\n[Step 2] Running rerun_all_ccd_shared982.py with env:")
    for k in ["CCD_HARD_NEG_JSONL", "CCD_EVAL_KEEP_MASK_NPY", "CCD_BOOTSTRAP", "CCD_K_NEG", "CCD_SEED"]:
        print(f"  {k}={env.get(k, '(unset)')}")
    
    cmd = ["python", str(PROJ / "tools" / "rerun_all_ccd_shared982.py")]
    result = subprocess.run(cmd, cwd=str(PROJ), env=env)
    
    print(f"\n[Step 2] Batch run complete. Return code: {result.returncode}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
