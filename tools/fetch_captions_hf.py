#!/usr/bin/env python3
"""Fetch COCO captions via HuggingFace datasets (avoids manual large zip download).

This script attempts several dataset/config combinations to retrieve 2017 COCO captions.
It gathers all caption strings from train + val splits and saves them to a .pt list[str].

Usage:
  python tools/fetch_captions_hf.py --out_pt data/coco_hf_captions.pt --out_txt data/coco_hf_captions.txt

Optional:
  --limit 50000  (use only first 50k captions for quick indexing)
  --lower        (lowercase captions)
  --dedup        (deduplicate case-insensitive)

You can then build an index:
  python tools/text_index.py --captions_pt data/coco_hf_captions.pt --out_pt data/coco_hf_index.pt --max_samples 50000
"""
import argparse, os, sys
import torch

TRY_DATASETS = [
    ("coco_captions", "2017"),  # common HF dataset naming
    ("HuggingFaceM4/COCO", "2017_captions"),
]

def attempt_load():
    from datasets import load_dataset
    for name, config in TRY_DATASETS:
        try:
            print(f"[try] dataset={name} config={config}")
            ds_train = load_dataset(name, config, split="train")
            # val split naming variations
            val_split_candidates = ["validation", "val"]
            ds_val = None
            for vs in val_split_candidates:
                try:
                    ds_val = load_dataset(name, config, split=vs)
                    break
                except Exception:
                    continue
            if ds_val is None:
                # fallback: skip val
                print("[warn] validation split not found; proceeding with train only")
                all_caps = [r.get("caption", "") for r in ds_train]
            else:
                all_caps = [r.get("caption", "") for r in ds_train] + [r.get("caption", "") for r in ds_val]
            return all_caps, name, config
        except Exception as e:
            print(f"[fail] {name}/{config}: {e}")
    raise RuntimeError("No suitable COCO captions dataset configuration succeeded.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--out_txt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # noqa
    except Exception:
        print("[error] 'datasets' library not installed. Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    caps, ds_name, config = attempt_load()
    print(f"[ok] loaded {len(caps)} captions from {ds_name}/{config}")

    # filter / transform
    processed = []
    for c in caps:
        if not isinstance(c, str):
            continue
        c2 = c.strip()
        if not c2:
            continue
        if args.lower:
            c2 = c2.lower()
        processed.append(c2)

    if args.dedup:
        seen = set()
        ded = []
        for c in processed:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            ded.append(c)
        print(f"[dedup] reduced {len(processed)} -> {len(ded)}")
        processed = ded

    if args.limit > 0:
        processed = processed[:args.limit]
        print(f"[limit] truncated to {len(processed)} captions")

    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
    torch.save(processed, args.out_pt)
    print(f"[save] pt -> {args.out_pt} (count={len(processed)})")
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
        with open(args.out_txt, 'w', encoding='utf-8') as f:
            for c in processed:
                f.write(c + '\n')
        print(f"[save] txt -> {args.out_txt}")

if __name__ == '__main__':
    main()
