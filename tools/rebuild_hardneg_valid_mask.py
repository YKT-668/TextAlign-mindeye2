#!/usr/bin/env python
# coding: utf-8
"""Rebuild hardneg coverage mask for shared982.

This script recomputes which shared982 images have at least one hard negative
entry in cache/hardneg/shared982_hardneg.jsonl.

Outputs:
- cache/hardneg/hardneg_valid_mask.npy (bool, len=982)
- cache/hardneg/hardneg_valid_ids.npy (int64, indices where mask True)
- cache/hardneg/coverage_report.json

Rationale:
- shared982 is a strict subset of shared1000. The mask must be defined in the
  canonical shared982 order (src/shared982.npy), not by shared1000 indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]

SHARED982_PATH = PROJ / "src" / "shared982.npy"
HARDNEG_JSONL = PROJ / "cache" / "hardneg" / "shared982_hardneg.jsonl"
OUT_MASK = PROJ / "cache" / "hardneg" / "hardneg_valid_mask.npy"
OUT_IDS = PROJ / "cache" / "hardneg" / "hardneg_valid_ids.npy"
OUT_REPORT = PROJ / "cache" / "hardneg" / "coverage_report.json"


def _summary_stats(x: np.ndarray) -> dict:
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0}
    x = x.astype(np.float64)
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
    }


def _load_all_captions() -> list[str] | None:
    """Best-effort load captions for missing-id length stats."""

    captions_path = PROJ / "evals" / "all_captions.pt"
    if not captions_path.is_file():
        return None
    try:
        import torch

        obj = torch.load(str(captions_path), map_location="cpu")
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, list):
            return [str(x) for x in obj]
        return None
    except Exception:
        return None


def _load_shared1000_global_ids() -> np.ndarray | None:
    p = PROJ / "src" / "shared1000.npy"
    if not p.is_file():
        return None
    try:
        m = np.load(p)
        if m.dtype == np.bool_ and m.ndim == 1:
            return np.where(m)[0].astype(np.int64)
        if m.ndim == 1:
            return m.astype(np.int64)
        return None
    except Exception:
        return None


def main() -> None:
    if not SHARED982_PATH.is_file():
        raise FileNotFoundError(f"Missing: {SHARED982_PATH}")
    if not HARDNEG_JSONL.is_file():
        raise FileNotFoundError(f"Missing: {HARDNEG_JSONL}")

    m982 = np.load(SHARED982_PATH)
    ids982 = np.where(m982)[0].astype(np.int64) if m982.dtype == np.bool_ else m982.astype(np.int64)
    ids982_set = set(map(int, ids982.tolist()))

    seen = set()
    type_counts = {}
    n_lines = 0

    with open(HARDNEG_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            image_id = int(row.get("image_id"))
            if image_id in ids982_set:
                seen.add(image_id)
            t = row.get("type", None)
            if t is not None:
                type_counts[str(t)] = int(type_counts.get(str(t), 0)) + 1

    mask = np.array([int(i) in seen for i in ids982.tolist()], dtype=np.bool_)
    valid_ids = np.where(mask)[0].astype(np.int64)

    missing_global_ids = ids982[np.where(~mask)[0]].astype(np.int64)

    captions = _load_all_captions()
    missing_caption_chars: list[int] = []
    missing_caption_words: list[int] = []
    missing_oob = 0
    missing_examples: list[dict] = []
    if captions is not None:
        missing_idx = np.where(~mask)[0].astype(np.int64)
        # Two supported formats:
        # A) len(captions)==982 and aligned to shared982 order
        # B) len(captions)>982 and indexed by global image_id
        # C) len(captions)==1000 aligned to shared1000 order (common in this repo)
        if len(captions) == int(ids982.shape[0]):
            for j in missing_idx.tolist():
                cap_s = str(captions[int(j)])
                missing_caption_chars.append(len(cap_s))
                missing_caption_words.append(len(cap_s.strip().split()))
            for j in missing_idx[:10].tolist():
                missing_examples.append({"shared982_index": int(j), "image_id": int(ids982[int(j)]), "caption": str(captions[int(j)])})
        elif len(captions) == 1000:
            shared1000_ids = _load_shared1000_global_ids()
            if shared1000_ids is not None and int(shared1000_ids.size) == 1000:
                id2idx = {int(g): int(i) for i, g in enumerate(shared1000_ids.tolist())}
                for gid in missing_global_ids.tolist():
                    if int(gid) not in id2idx:
                        missing_oob += 1
                        continue
                    cap_s = str(captions[id2idx[int(gid)]])
                    missing_caption_chars.append(len(cap_s))
                    missing_caption_words.append(len(cap_s.strip().split()))
                for gid in missing_global_ids[:10].tolist():
                    if int(gid) in id2idx:
                        missing_examples.append({"image_id": int(gid), "caption": str(captions[id2idx[int(gid)]])})
            else:
                missing_oob = int(missing_global_ids.size)
        else:
            for gid in missing_global_ids.tolist():
                if gid < 0 or gid >= len(captions):
                    missing_oob += 1
                    continue
                cap_s = str(captions[int(gid)])
                missing_caption_chars.append(len(cap_s))
                missing_caption_words.append(len(cap_s.strip().split()))
            for gid in missing_global_ids[:10].tolist():
                if 0 <= int(gid) < len(captions):
                    missing_examples.append({"image_id": int(gid), "caption": str(captions[int(gid)])})

    OUT_MASK.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_MASK, mask)
    np.save(OUT_IDS, valid_ids)

    report = {
        "shared982_size": int(ids982.shape[0]),
        "hardneg_jsonl_lines": int(n_lines),
        "unique_image_id_in_jsonl": int(len(seen)),
        "n_valid": int(mask.sum()),
        "coverage_pct": float(mask.mean() * 100.0),
        "n_missing": int((~mask).sum()),
        "missing_caption_stats": {
            "captions_path": str((PROJ / "evals" / "all_captions.pt").resolve()),
            "missing_global_ids": {
                "count": int(missing_global_ids.size),
                "note": "IDs are global image_id indices aligned to evals/all_captions.pt",
            },
            "captions_format": (
                "shared982_order"
                if (captions is not None and len(captions) == int(ids982.shape[0]))
                else ("shared1000_order" if (captions is not None and len(captions) == 1000) else "global_ids")
            ),
            "length_chars": _summary_stats(np.asarray(missing_caption_chars, dtype=np.int64)),
            "length_words": _summary_stats(np.asarray(missing_caption_words, dtype=np.int64)),
            "out_of_bounds_ids": int(missing_oob),
            "examples": missing_examples,
        },
        "type_distribution_lines": type_counts,
        "paths": {
            "shared982": str(SHARED982_PATH),
            "hardneg_jsonl": str(HARDNEG_JSONL),
            "valid_mask": str(OUT_MASK),
            "valid_ids": str(OUT_IDS),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] shared982={int(ids982.shape[0])} covered={int(mask.sum())} ({report['coverage_pct']:.2f}%)")
    print(f"[OK] wrote {OUT_MASK}")
    print(f"[OK] wrote {OUT_IDS}")
    print(f"[OK] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
