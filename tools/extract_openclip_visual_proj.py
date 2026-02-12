#!/usr/bin/env python
# coding: utf-8
"""Extract OpenCLIP visual.proj for ViT-bigG-14.

This generates a cached projection matrix to make CCD evaluation offline-friendly.

Usage:
  python tools/extract_openclip_visual_proj.py \
    --model ViT-bigG-14 --pretrained laion2b_s39b_b160k \
    --out cache/model_eval_results/shared982_ccd_assets/openclip_visual_proj_ViT-bigG-14_laion2b_s39b_b160k.pt

Requires:
  pip install open-clip-torch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-bigG-14")
    ap.add_argument("--pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import open_clip
    except Exception as e:
        raise SystemExit(
            "[FATAL] open_clip not found. Install with `pip install open-clip-torch` (or provide the proj file manually).\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    proj = getattr(model.visual, "proj", None)
    if proj is None:
        raise SystemExit("[FATAL] model.visual.proj is missing")

    proj_t = proj.detach().cpu().float()
    if proj_t.ndim != 2:
        raise SystemExit(f"[FATAL] Unexpected proj shape: {tuple(proj_t.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj_t, out)
    print(f"[OK] Saved visual.proj shape={tuple(proj_t.shape)} to {out}")


if __name__ == "__main__":
    main()
