#!/usr/bin/env python3
"""Quick RAG feature-store quality checker.

Usage:
  python tools/check_rag_store.py --text_index data/text_index_vith.pt --captions data/all_captions.pt \
       --num_samples 50 --k 5 --out runs/rag_check.json

What it reports:
 - index shape, dtype
 - norm statistics (L2 norms)
 - fraction of near-duplicates (cosine > 0.999)
 - sampled k-NN stats (mean top1/top5 cosine excluding self)
 - for a few samples, prints sample caption + top-k captions for human inspection

This script avoids external deps beyond torch/numpy.
"""
import argparse
import json
import random
import os
from typing import Any

import torch
import numpy as np


def load_any(path: str) -> Any:
    obj = torch.load(path, map_location='cpu')
    return obj


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    norms = x.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    return x / norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text_index', required=True)
    ap.add_argument('--captions', required=False, default='')
    ap.add_argument('--num_samples', type=int, default=50)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='runs/rag_check.json')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    report = {'text_index_path': args.text_index, 'captions_path': args.captions}

    print(f"Loading text index from {args.text_index}...")
    ti = load_any(args.text_index)
    # Unwrap common containers
    if isinstance(ti, dict):
        # common key names
        for k in ('embeds', 'vectors', 'features', 'X', 'T'):
            if k in ti:
                ti = ti[k]
                break
        else:
            # If dict of numpy arrays
            vals = list(ti.values())
            if len(vals) == 1:
                ti = vals[0]

    if isinstance(ti, list):
        ti = np.asarray(ti)

    if isinstance(ti, np.ndarray):
        vecs = torch.from_numpy(ti)
    elif isinstance(ti, torch.Tensor):
        vecs = ti
    else:
        raise RuntimeError(f'Unsupported text_index type: {type(ti)}')

    vecs = vecs.float()
    N, D = vecs.shape
    report.update({'N': int(N), 'D': int(D), 'dtype': str(vecs.dtype)})
    print(f"Index shape: {vecs.shape}, dtype={vecs.dtype}")

    norms = vecs.norm(dim=1).numpy()
    report['norms'] = {
        'min': float(norms.min()), 'max': float(norms.max()), 'mean': float(norms.mean()), 'std': float(norms.std())
    }
    print("Norms:", report['norms'])

    # Duplicate / near-duplicate check: sample up to 2000 pairs
    m = min(2000, N)
    idxs = np.random.choice(N, size=m, replace=False)
    samp = vecs[idxs]
    samp_n = normalize_rows(samp)
    all_n = normalize_rows(vecs)
    # compute pairwise similarities (samp x all) in batches to avoid OOM
    batch = 256
    near_dup_count = 0
    total_pairs = 0
    # When checking near-duplicates, exclude the trivial self-match by masking it out
    for i in range(0, m, batch):
        b = samp_n[i:i+batch]
        sims = b @ all_n.t()
        for row_idx in range(sims.shape[0]):
            gi = int(idxs[i + row_idx])
            # mask self similarity so we inspect the nearest *other* vector
            sims[row_idx, gi] = -9e9
            total_pairs += 1
            top1 = sims[row_idx].max().item()
            if top1 > 0.9999:
                near_dup_count += 1

    report['near_dup_fraction'] = float(near_dup_count) / float(total_pairs) if total_pairs > 0 else 0.0
    print(f"Near-duplicate fraction (sampled): {report['near_dup_fraction']:.6f}")

    # k-NN stats on sampled queries
    num_q = min(args.num_samples, N)
    q_idx = np.random.choice(N, size=num_q, replace=False)
    qs = vecs[q_idx]
    qs_n = normalize_rows(qs)
    all_n = normalize_rows(vecs)

    top1_list = []
    topk_mean_list = []
    topk_indices = []
    for i in range(0, num_q, 64):
        b = qs_n[i:i+64]
        sims = b @ all_n.t()
        # set self-sim to -inf so we exclude trivial self match
        for j, gi in enumerate(q_idx[i:i+64]):
            sims[j, gi] = -9e9
        topk = sims.topk(min(args.k, sims.shape[1]), dim=1)
        top_vals = topk.values.cpu().numpy()
        top_idx = topk.indices.cpu().numpy()
        for r in range(top_vals.shape[0]):
            top1_list.append(float(top_vals[r, 0]))
            topk_mean_list.append(float(top_vals[r, :min(args.k, top_vals.shape[1])].mean()))
            topk_indices.append(top_idx[r].tolist())

    report['knn'] = {
        'k': int(args.k),
        'num_queries': int(num_q),
        'mean_top1_cosine': float(np.mean(top1_list)),
        'mean_topk_cosine': float(np.mean(topk_mean_list))
    }
    print("k-NN summary:", report['knn'])

    # Load captions if provided
    captions = None
    if args.captions and os.path.isfile(args.captions):
        print(f"Loading captions from {args.captions}...")
        cap_obj = load_any(args.captions)
        # try to unwrap
        if isinstance(cap_obj, dict):
            # try common keys
            for k in ('captions', 'texts', 'all_captions'):
                if k in cap_obj:
                    captions = cap_obj[k]
                    break
            else:
                # if dict of lists, try values
                vals = list(cap_obj.values())
                if len(vals) == N:
                    captions = vals
        elif isinstance(cap_obj, list):
            captions = cap_obj
        elif isinstance(cap_obj, np.ndarray):
            captions = cap_obj.tolist()

        if captions is None:
            print('Warning: could not parse captions structure; will skip captions display')
        else:
            if len(captions) != N:
                print(f'Warning: captions length ({len(captions)}) != index size ({N})')
            report['captions_sample'] = []

            # show a few example queries and their topk captions
            show_k = min(5, num_q)
            for qi, topk_idx in zip(q_idx[:show_k], topk_indices[:show_k]):
                qcap = captions[qi] if qi < len(captions) else ''
                top_caps = [captions[ii] if ii < len(captions) else '' for ii in topk_idx[:args.k]]
                entry = {'query_index': int(qi), 'query_caption': qcap, 'topk_idx': [int(x) for x in topk_idx[:args.k]], 'topk_captions': top_caps}
                report['captions_sample'].append(entry)

    # Save report
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to {out_path}")


if __name__ == '__main__':
    main()
