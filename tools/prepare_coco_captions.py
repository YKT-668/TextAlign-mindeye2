#!/usr/bin/env python3
"""Prepare a large caption corpus from COCO-style annotation JSON(s).

This converts one or more COCO caption JSON files (e.g. captions_train2017.json,
captions_val2017.json) into:
  1) A deduplicated list of raw caption strings saved as a .pt (for text_index.py)
  2) (Optional) A .txt file (one caption per line) for inspection / other tooling.

Example:
  python tools/prepare_coco_captions.py \
      --ann captions_train2017.json captions_val2017.json \
      --out_pt data/all_coco_captions.pt \
      --out_txt data/all_coco_captions.txt \
      --min_len 3 --max_len 140

Then build the index:
  python tools/text_index.py --captions_pt data/all_coco_captions.pt --out_pt data/text_index_vith_large.pt

The resulting PT file is a list[str]. Existing text_index.py already supports list of str.
"""
import argparse, json, os, re, sys
from typing import List
import torch

def load_coco_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    if 'annotations' not in j:
        raise ValueError(f"File {path} does not look like a COCO captions JSON (missing 'annotations').")
    caps = []
    for ann in j['annotations']:
        c = ann.get('caption', '')
        if not isinstance(c, str):
            continue
        caps.append(c.strip())
    return caps

def clean_caption(c: str) -> str:
    # Basic normalization: collapse whitespace, remove leading/trailing punctuation spaces
    c = re.sub(r"\s+", " ", c).strip()
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', nargs='+', required=True, help='One or more COCO caption JSON files.')
    ap.add_argument('--out_pt', required=True, help='Output .pt path (list[str])')
    ap.add_argument('--out_txt', default='', help='Optional .txt (one caption per line)')
    ap.add_argument('--min_len', type=int, default=1, help='Drop captions shorter than this (after strip).')
    ap.add_argument('--max_len', type=int, default=300, help='Truncate captions longer than this.')
    ap.add_argument('--dedup', action='store_true', help='Enable global deduplication (case-insensitive).')
    ap.add_argument('--lower', action='store_true', help='Convert to lowercase for more uniform indexing.')
    args = ap.parse_args()

    all_caps: List[str] = []
    for p in args.ann:
        if not os.path.isfile(p):
            print(f"[warn] File not found: {p}", file=sys.stderr)
            continue
        caps = load_coco_file(p)
        print(f"[load] {p}: {len(caps)} captions")
        all_caps.extend(caps)

    print(f"[merge] total raw captions: {len(all_caps)}")

    cleaned: List[str] = []
    for c in all_caps:
        c2 = clean_caption(c)
        if args.lower:
            c2 = c2.lower()
        if len(c2) < args.min_len:
            continue
        if len(c2) > args.max_len:
            c2 = c2[:args.max_len].rstrip()
        cleaned.append(c2)

    print(f"[filter] after len constraints: {len(cleaned)}")

    if args.dedup:
        seen = set()
        deduped = []
        for c in cleaned:
            key = c.lower() if args.lower else c.lower()  # case-insensitive dedup always
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        print(f"[dedup] unique captions: {len(deduped)} (removed {len(cleaned)-len(deduped)})")
        cleaned = deduped

    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
    torch.save(cleaned, args.out_pt)
    print(f"[save] .pt -> {args.out_pt} (list[str], count={len(cleaned)})")

    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
        with open(args.out_txt, 'w', encoding='utf-8') as f:
            for c in cleaned:
                f.write(c + '\n')
        print(f"[save] .txt -> {args.out_txt}")

    # Tiny summary stats
    lengths = [len(c) for c in cleaned]
    if lengths:
        import numpy as np
        arr = np.array(lengths)
        print(f"[stats] caption length chars: min={arr.min()} mean={arr.mean():.1f} p90={np.percentile(arr,90):.0f} max={arr.max()}")

if __name__ == '__main__':
    main()
