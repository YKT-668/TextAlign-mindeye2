#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Tuple

import torch


def _load_state_dict(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    meta: Dict[str, Any] = {"path": path}

    if isinstance(obj, dict):
        meta["top_keys"] = list(obj.keys())
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            meta["format"] = "lightning"
            return obj["state_dict"], meta
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            meta["format"] = "mindeye_ckpt"
            return obj["model_state_dict"], meta
        if all(isinstance(k, str) for k in obj.keys()):
            # raw state_dict
            meta["format"] = "raw_state_dict"
            return obj, meta

    raise RuntimeError(f"Unsupported checkpoint format at {path}. type={type(obj)}")


def _filter_keys(keys, keywords):
    out = []
    for k in sorted(keys):
        lk = k.lower()
        if any(kw in lk for kw in keywords):
            out.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--keywords", default="text,align,clip_proj,backbone_linear,text_head,ridge")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    ckpt_a = os.path.expanduser(args.ckpt_a)
    ckpt_b = os.path.expanduser(args.ckpt_b)
    keywords = [x.strip().lower() for x in args.keywords.split(",") if x.strip()]

    sd_a, meta_a = _load_state_dict(ckpt_a)
    sd_b, meta_b = _load_state_dict(ckpt_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    inter = sorted(keys_a & keys_b)

    kw_a = _filter_keys(keys_a, keywords)
    kw_b = _filter_keys(keys_b, keywords)

    def _summ(sd: Dict[str, torch.Tensor]):
        dtypes = {}
        for t in sd.values():
            if isinstance(t, torch.Tensor):
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
        return {
            "num_keys": len(sd),
            "dtype_hist": dtypes,
        }

    summary = {
        "a": {"meta": meta_a, **_summ(sd_a)},
        "b": {"meta": meta_b, **_summ(sd_b)},
        "intersection": len(inter),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "keyword_matches": {"keywords": keywords, "a": kw_a, "b": kw_b},
        "diff": {"a_minus_b": only_a, "b_minus_a": only_b},
    }

    print("=== CKPT KEY DIFF ===")
    print(f"A: {ckpt_a}")
    print(f"  format={meta_a.get('format')} top_keys={meta_a.get('top_keys')}")
    print(f"  num_keys={len(sd_a)}")
    print(f"B: {ckpt_b}")
    print(f"  format={meta_b.get('format')} top_keys={meta_b.get('top_keys')}")
    print(f"  num_keys={len(sd_b)}")
    print()
    print(f"intersection: {len(inter)}")
    print(f"only_in_A: {len(only_a)}")
    print(f"only_in_B: {len(only_b)}")
    print()
    print("--- Keyword keys (A) ---")
    for k in kw_a:
        print(k)
    print("--- Keyword keys (B) ---")
    for k in kw_b:
        print(k)
    print()
    print("--- A - B (first 200) ---")
    for k in only_a[:200]:
        print(k)
    if len(only_a) > 200:
        print(f"... ({len(only_a)-200} more)")
    print("--- B - A (first 200) ---")
    for k in only_b[:200]:
        print(k)
    if len(only_b) > 200:
        print(f"... ({len(only_b)-200} more)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[WROTE] {args.json_out}")


if __name__ == "__main__":
    main()
