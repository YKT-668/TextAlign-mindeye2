#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError(f"Unsupported checkpoint format at {path} (type={type(obj)})")


def _pick_key(sd: Dict[str, torch.Tensor], candidates) -> str:
    for k in candidates:
        if k in sd:
            return k
    # fallback: try substring
    for k in sd.keys():
        lk = k.lower()
        if any(c.lower() in lk for c in candidates):
            return k
    raise KeyError(f"None of candidates found: {candidates}")


def _tensor_stats(x: torch.Tensor):
    x = x.detach().float().view(-1)
    return {
        "abs_mean": float(x.abs().mean().item()),
        "l2_norm": float(x.norm(p=2).item()),
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor):
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    d = b - a
    cos = float(F.cosine_similarity(a, b, dim=0).item()) if a.numel() == b.numel() else float("nan")
    return {
        "delta_l2": float(d.norm(p=2).item()),
        "cosine": cos,
        "a": _tensor_stats(a),
        "b": _tensor_stats(b),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_end", required=True)
    ap.add_argument("--ckpt_start", default=None, help="Optional. Start checkpoint. If omitted, use --ckpt_ref as baseline.")
    ap.add_argument("--ckpt_ref", default=None, help="Optional. Reference checkpoint (e.g., v2 last). Used if --ckpt_start missing.")
    args = ap.parse_args()

    end_path = os.path.expanduser(args.ckpt_end)
    start_path = os.path.expanduser(args.ckpt_start) if args.ckpt_start else None
    ref_path = os.path.expanduser(args.ckpt_ref) if args.ckpt_ref else None

    if start_path is None:
        if ref_path is None:
            raise SystemExit("Need either --ckpt_start or --ckpt_ref")
        start_path = ref_path

    sd_start = _load_state_dict(start_path)
    sd_end = _load_state_dict(end_path)

    # representative tensors per requirement:
    # - TextAlign head / clip_proj weight
    # - backbone_linear weight
    # - any bias
    key_text_head_w = _pick_key(sd_end, [
        "text_head.mlp.1.weight",
        "text_head.mlp.3.weight",
        "model.text_head.mlp.1.weight",
        "model.text_head.mlp.3.weight",
        "text_head",
    ])
    key_backbone_linear_w = _pick_key(sd_end, [
        "backbone.backbone_linear.weight",
        "backbone_linear.weight",
        "model.backbone.backbone_linear.weight",
        "backbone.backbone_linear",
        "backbone_linear",
    ])
    # bias: prefer text_head bias
    key_bias = _pick_key(sd_end, [
        "text_head.mlp.3.bias",
        "text_head.mlp.1.bias",
        "backbone.backbone_linear.bias",
        "ridge.linears.0.bias",
        "bias",
    ])

    keys = [
        ("textalign_head_weight", key_text_head_w),
        ("backbone_linear_weight", key_backbone_linear_w),
        ("some_bias", key_bias),
    ]

    print("=== WEIGHT UPDATE PROBE ===")
    print(f"start: {start_path}")
    print(f"end  : {end_path}")
    print()

    for tag, k in keys:
        if k not in sd_start:
            print(f"[{tag}] key={k} not in start ckpt; skipping")
            continue
        if k not in sd_end:
            print(f"[{tag}] key={k} not in end ckpt; skipping")
            continue
        a = sd_start[k]
        b = sd_end[k]
        if a.shape != b.shape:
            print(f"[{tag}] key={k} shape mismatch start={tuple(a.shape)} end={tuple(b.shape)}")
            continue
        st = _diff_stats(a, b)
        print(f"[{tag}] key={k} shape={tuple(a.shape)}")
        print(f"  delta_l2={st['delta_l2']}")
        print(f"  cosine  ={st['cosine']}")
        print(f"  start abs_mean={st['a']['abs_mean']} l2_norm={st['a']['l2_norm']}")
        print(f"  end   abs_mean={st['b']['abs_mean']} l2_norm={st['b']['l2_norm']}")
        print()


if __name__ == "__main__":
    main()
