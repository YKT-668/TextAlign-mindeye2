#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Series:
    name: str
    steps: List[int]
    values: List[float]


_KV_RE = re.compile(r"(?P<k>train/[^=,\s]+)=(?P<v>-?\d+(?:\.\d+)?(?:e-?\d+)?)")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_progress_kvs(text: str) -> Dict[str, Series]:
    # Parse lines that include "train/num_steps=" (tqdm postfix) and key=value pairs.
    out: Dict[str, Series] = {}
    for line in text.splitlines():
        if "train/num_steps=" not in line:
            continue
        # find global step
        m_step = re.search(r"train/num_steps=(\d+)", line)
        if not m_step:
            continue
        step = int(m_step.group(1))
        for m in _KV_RE.finditer(line):
            k = m.group("k")
            v = float(m.group("v"))
            if k not in out:
                out[k] = Series(k, [], [])
            out[k].steps.append(step)
            out[k].values.append(v)
    return out


def _extract_epoch_step_loss_lr(text: str) -> Tuple[List[int], List[float], List[float]]:
    # Parse explicit prints like: [epoch 0/10 step 50/937] loss=3.6537 lr=1.25e-05
    steps: List[int] = []
    losses: List[float] = []
    lrs: List[float] = []
    re_line = re.compile(
        r"\[epoch\s+(\d+)/(\d+)\s+step\s+(\d+)/(\d+)\]\s+loss=(?P<loss>-?\d+(?:\.\d+)?)\s+lr=(?P<lr>-?\d+(?:\.\d+)?(?:e-?\d+)?)"
    )
    for line in text.splitlines():
        m = re_line.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        in_epoch_step = int(m.group(3))
        steps.append(epoch * 1_000_000 + in_epoch_step)  # unique ordering; real global step handled elsewhere
        losses.append(float(m.group("loss")))
        lrs.append(float(m.group("lr")))
    return steps, losses, lrs


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "min": min(values),
        "max": max(values),
    }


def _window_by_steps(steps: List[int], values: List[float], first_k: int, last_k: int) -> Dict[str, Dict[str, float]]:
    if not steps:
        return {"first": {"n": 0}, "last": {"n": 0}}
    pairs = sorted(zip(steps, values), key=lambda x: x[0])
    values_sorted = [v for _, v in pairs]
    first = values_sorted[: min(first_k, len(values_sorted))]
    last = values_sorted[-min(last_k, len(values_sorted)) :]
    return {"first": _stats(first), "last": _stats(last)}


def _resolve_one(path_glob: str) -> str:
    paths = sorted(glob.glob(path_glob))
    if not paths:
        raise FileNotFoundError(f"No log matched: {path_glob}")
    if len(paths) > 1:
        # choose the largest (usually the full log)
        paths = sorted(paths, key=lambda p: os.path.getsize(p), reverse=True)
    return paths[0]


def summarize_log(tag: str, path: str) -> None:
    text = _read_text(path)
    kvs = _extract_progress_kvs(text)
    print(f"=== {tag} ===")
    print(f"log: {path}")
    want = ["train/loss_text", "train/loss", "train/lr", "train/loss_clip_total", "train/loss_prior"]
    for k in want:
        if k not in kvs:
            print(f"- {k}: NOT FOUND")
            continue
        s = kvs[k]
        w = _window_by_steps(s.steps, s.values, first_k=2000, last_k=2000)
        print(f"- {k}: total_points={len(s.values)}")
        print(f"  first2000: n={w['first'].get('n')} mean={w['first'].get('mean')} var={w['first'].get('var')} min={w['first'].get('min')} max={w['first'].get('max')}")
        print(f"  last2000 : n={w['last'].get('n')} mean={w['last'].get('mean')} var={w['last'].get('var')} min={w['last'].get('min')} max={w['last'].get('max')}")
        # sample head/tail 20
        pairs = sorted(zip(s.steps, s.values), key=lambda x: x[0])
        head = pairs[:20]
        tail = pairs[-20:]
        print("  head20:")
        for st, v in head:
            print(f"    step={st} value={v}")
        print("  tail20:")
        for st, v in tail:
            print(f"    step={st} value={v}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_v2", required=True)
    ap.add_argument("--log_v3", required=True)
    args = ap.parse_args()

    v2 = _resolve_one(args.log_v2)
    v3 = _resolve_one(args.log_v3)
    summarize_log("v2", v2)
    summarize_log("v3", v3)


if __name__ == "__main__":
    main()
