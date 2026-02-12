#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds


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


def _infer_hidden_dim_nblocks(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    # hidden_dim from ridge weight: [hidden_dim, n_vox]
    hid = None
    for k, t in sd.items():
        if k.endswith("ridge.linears.0.weight") and isinstance(t, torch.Tensor) and t.ndim == 2:
            hid = int(t.shape[0])
            break
    if hid is None:
        raise RuntimeError("Could not infer hidden_dim from ridge.linears.0.weight")
    # n_blocks from backbone.mixer_blocks1.N
    n_blocks = 0
    for k in sd.keys():
        if k.startswith("backbone.mixer_blocks1."):
            try:
                idx = int(k.split(".")[2])
                n_blocks = max(n_blocks, idx + 1)
            except Exception:
                continue
    if n_blocks <= 0:
        n_blocks = 4
    return hid, n_blocks


def _pick_param(model: torch.nn.Module, candidates: List[str]):
    named = dict(model.named_parameters())
    for k in candidates:
        if k in named:
            return k, named[k]
    # substring fallback
    for name, p in named.items():
        lname = name.lower()
        if any(c.lower() in lname for c in candidates):
            return name, p
    raise KeyError(f"No param found for candidates: {candidates}")


def text_align_loss(t_pred, t_teacher, tau=0.07):
    t_pred = F.normalize(t_pred, dim=-1)
    t_teacher = F.normalize(t_teacher, dim=-1)
    logits = t_pred @ t_teacher.t() / tau
    labels = torch.arange(t_pred.size(0), device=t_pred.device)
    return F.cross_entropy(logits, labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_path", required=True, help="Repo src root (contains wds/ and betas_*.hdf5)")
    ap.add_argument("--ckpt", default=None, help="Optional. If omitted, resolve from /mnt/work/repos/textalign-mindeye2-model/models/subj01/<model_name>/last.pth")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size; default uses log-consistent heuristic (32 for v3 short).")
    args = ap.parse_args()

    subj = int(args.subj)
    model_name = args.model_name
    data_path = os.path.abspath(args.data_path)

    if args.ckpt:
        ckpt_path = os.path.expanduser(args.ckpt)
    else:
        ckpt_path = f"/mnt/work/repos/textalign-mindeye2-model/models/subj0{subj:02d}/{model_name}/last.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    # Match Train_textalign.py behavior: TextAlign uses teacher feats and id2row
    teacher_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/nsd_text/train_coco_text_clip.pt")
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(f"teacher feats missing: {teacher_path}")
    state = torch.load(teacher_path, map_location="cpu")
    image_ids = state["image_ids"].long().tolist()
    id2row = {int(i): idx for idx, i in enumerate(image_ids)}
    text_feats_teacher = state["text_feats"].float()  # keep on CPU for now

    # Ensure repo root and src/ are importable (Train_textalign.py expects src/ on sys.path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Build minimal model identical modules used in Train_textalign.py
    from models_textalign import BrainNetwork, TextAlignHead  # type: ignore

    sd = _load_state_dict(ckpt_path)
    hidden_dim, n_blocks = _infer_hidden_dim_nblocks(sd)

    n_vox = None
    w = sd.get("ridge.linears.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        n_vox = int(w.shape[1])
    if n_vox is None:
        raise RuntimeError("Could not infer n_vox")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class RidgeRegression(torch.nn.Module):
        def __init__(self, input_sizes, out_features):
            super().__init__()
            self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])

        def forward(self, x, subj_idx):
            return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

    class MindEyeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

    model = MindEyeModule()
    model.ridge = RidgeRegression([n_vox], out_features=hidden_dim)
    model.backbone = BrainNetwork(
        h=hidden_dim,
        in_dim=hidden_dim,
        seq_len=1,
        n_blocks=n_blocks,
        clip_size=1664,
        out_dim=1664 * 256,
        blurry_recon=False,
        clip_scale=1.0,
    )
    model.text_head = TextAlignHead(token_dim=1664, hidden_dim=2048, text_dim=768)

    # Freeze like Train_textalign.py (only text_head trainable)
    for p in model.ridge.parameters():
        p.requires_grad_(False)
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    # Load ckpt with strict=False because repo has known naming divergences elsewhere
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model.to(device)
    model.train()
    text_feats_teacher = text_feats_teacher.to(device)

    # Prepare one batch from training wds to get behav fields
    tar = os.path.join(data_path, f"wds/subj0{subj}/train/0.tar")
    if not os.path.isfile(tar):
        raise FileNotFoundError(f"train tar missing: {tar}")

    dataset = (
        wds.WebDataset(tar, resampled=False)
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    bs = int(args.batch_size) if args.batch_size is not None else 32
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)
    behav0, past_behav0, future_behav0, old_behav0 = next(iter(dl))

    image_idx = behav0[:, 0, 0].cpu().long().numpy()
    image0, image_sorted_idx = np.unique(image_idx, return_index=True)
    if len(image0) != len(image_idx):
        # if duplicates, follow Train_textalign.py: keep only unique samples by selecting indices
        behav0 = behav0[image_sorted_idx]
        image0 = image_idx[image_sorted_idx]

    global_ids = torch.from_numpy(np.asarray(image0, dtype=np.int64)).to(device)
    rows = torch.tensor([id2row.get(int(gid), -1) for gid in global_ids.detach().cpu().tolist()], device=device)
    valid_mask = rows >= 0

    alpha_text = float(os.environ.get("MINDEYE_TEXTALIGN_SCALE", "0.05"))
    use_textalign_env = os.environ.get("MINDEYE_TEXTALIGN", "1") == "1"
    use_textalign_flag = bool(use_textalign_env) and alpha_text > 0

    print("=== ONE BATCH GRADCHECK (no optimizer.step) ===")
    print(f"ckpt: {ckpt_path}")
    print(f"device: {device}")
    print(f"batch_size(raw)={bs} unique_after_dedup={int(global_ids.shape[0])}")
    print(f"use_textalign_env={use_textalign_env} alpha_text={alpha_text}")
    print(f"shared_hits={int(valid_mask.sum().item())} / {int(valid_mask.numel())}")
    print(f"USE_TEXT_ALIGN(teacher_present)={True}")
    print()

    if (not use_textalign_flag) or (not valid_mask.any()):
        print("[SKIP] TextAlign loss branch would be skipped.")
        print(f"  reason: use_textalign_flag={use_textalign_flag} valid_any={bool(valid_mask.any())}")
        print("  Suggest: ensure MINDEYE_TEXTALIGN=1, MINDEYE_TEXTALIGN_SCALE>0, and batch includes shared1000 ids")
        return

    rows_valid = rows[valid_mask]
    t_pos = text_feats_teacher[rows_valid]

    # Need backbone tokens; ridge/backbone are frozen but forward is required.
    # For voxels, Train_textalign.py uses h5 betas via behav voxel_idx; for gradcheck, we only need a forward path that reaches text_head.
    # We'll create a fake ridge output to avoid h5 dependency.
    Bv = int(rows_valid.shape[0])
    # Fake backbone tokens by running backbone on random ridge outputs (no grad needed on backbone params anyway).
    voxel_ridge = torch.randn((Bv, 1, hidden_dim), device=device)
    with torch.no_grad():
        backbone_tokens, _, _ = model.backbone(voxel_ridge)

    t_pred = model.text_head(backbone_tokens)
    loss_text = text_align_loss(t_pred, t_pos, tau=0.07)
    loss_total = alpha_text * loss_text

    print(f"loss_text={float(loss_text.detach().item())}")
    print(f"loss_total={float(loss_total.detach().item())}")

    loss_total.backward()

    # pick representative parameters
    p1_name, p1 = _pick_param(model, ["text_head.mlp.1.weight", "text_head.mlp.3.weight", "text_head"])  # weight
    p2_name, p2 = _pick_param(model, ["text_head.mlp.3.bias", "text_head.mlp.1.bias", "bias"])  # bias
    # backbone_linear is frozen; grad should be None or 0 because requires_grad=False
    bp_name = None
    try:
        bp_name, bp = _pick_param(model, ["backbone.backbone_linear.weight", "backbone_linear.weight"])
    except Exception:
        bp = None

    def gstat(p: torch.nn.Parameter):
        if p.grad is None:
            return "grad=None"
        g = p.grad.detach()
        return f"grad_abs_mean={float(g.abs().mean().item())} grad_l2={float(g.norm(p=2).item())}"

    print("\n--- Grad stats ---")
    print(f"{p1_name}: requires_grad={p1.requires_grad} {gstat(p1)}")
    print(f"{p2_name}: requires_grad={p2.requires_grad} {gstat(p2)}")
    if bp_name is not None and bp is not None:
        print(f"{bp_name}: requires_grad={bp.requires_grad} {gstat(bp)}")


if __name__ == "__main__":
    # Ensure repo root import works when running from repo root
    if os.getcwd().endswith("/tools"):
        os.chdir(os.path.dirname(os.getcwd()))
    main()
