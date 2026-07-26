#!/usr/bin/env python3
"""verify_restored_archive.py - Verify restored ConceptAlign Final14 archive"""
import sys, os, hashlib, subprocess, struct
from pathlib import Path

ARCHIVE_DIR = Path(__file__).parent.parent
RESULTS = {}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_file(path, desc):
    if path.exists():
        sz = path.stat().st_size
        RESULTS[desc] = f"PASS ({sz} bytes)"
        return True
    RESULTS[desc] = "MISSING"
    return False

def check_npy(path, desc):
    if not path.exists():
        RESULTS[desc] = "MISSING"
        return False
    try:
        import numpy as np
        mm = np.load(path, mmap_mode='r')
        RESULTS[desc] = f"PASS (shape={mm.shape}, dtype={mm.dtype})"
        return True
    except Exception as e:
        RESULTS[desc] = f"FAIL: {e}"
        return False

def check_pth(path, desc):
    if not path.exists():
        RESULTS[desc] = "MISSING"
        return False
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        RESULTS[desc] = f"PASS ({len(keys)} keys, type={type(ckpt).__name__})"
        return True
    except Exception as e:
        RESULTS[desc] = f"FAIL: {e}"
        return False

def main():
    print("=== ConceptAlign Final14 Restore Verification ===\n")
    all_ok = True
    
    # Check manifests
    check_file(ARCHIVE_DIR / "manifests/CHECKSUMS.sha256", "checksums")
    check_file(ARCHIVE_DIR / "manifests/LOCAL_ASSET_INVENTORY.tsv", "inventory")
    check_file(ARCHIVE_DIR / "manifests/THIRD_PARTY_DEPENDENCIES.tsv", "deps")
    check_file(ARCHIVE_DIR / "manifests/DATA_PATH_MAP.tsv", "pathmap")
    
    # Check environment
    check_file(ARCHIVE_DIR / "reproducibility/environment/environment.yml", "env_yml")
    check_file(ARCHIVE_DIR / "reproducibility/environment/pip-freeze.txt", "pip_freeze")
    check_file(ARCHIVE_DIR / "reproducibility/environment/git-state.txt", "git_state")
    
    # Check restore scripts
    check_file(ARCHIVE_DIR / "restore/restore_from_cloud.sh", "restore_script")
    
    # Check readme
    root = ARCHIVE_DIR.parent
    check_file(root / "readme5-1.md", "readme5-1")
    
    # Check for checkpoints in local repo
    for ckpt in root.glob("train_logs/*/last.pth"):
        check_pth(ckpt, f"checkpoint_{ckpt.parent.name}")
    
    for ckpt in root.glob("mindeyev2_ckpts/**/last.pth"):
        check_pth(ckpt, f"mindeye2_ckpt_{ckpt.parent.name}")
    
    # Check conda-pack
    env_pack = root / "hf_model/final14_archive/reproducibility/conceptalign_env_linux64.tar.zst"
    if env_pack.exists():
        check_file(env_pack, "conda-pack")
    
    # Summary
    print("\n=== Results ===")
    for k, v in RESULTS.items():
        status = "✅" if "PASS" in str(v) else "❌"
        print(f"  {status} {k}: {v}")
    
    failed = [k for k, v in RESULTS.items() if "PASS" not in str(v)]
    if failed:
        print(f"\n❌ {len(failed)} checks failed")
        sys.exit(1)
    print(f"\n✅ All {len(RESULTS)} checks passed")
    sys.exit(0)

if __name__ == "__main__":
    main()