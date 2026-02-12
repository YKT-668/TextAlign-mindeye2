# Release Audit Report
**Date:** 2026-02-12
**Repository:** `/mnt/work/repos/TextAlign-mindeye2`

---

## 1. Git Status
- **Branch:** `main` (Ahead of `origin/main` by 1 commit)
- **Status:** **DIRTY**. There are uncommitted changes and many untracked files.
- **Recent History:**
  - `aad3b84` save updates before switching instance
  - `7bddf84` Add environment-mindeye.yml and update readme2
  - `a27351b` Update readme and add data links
  - `f23bed8` Add environment description for mindeye21 (conda + pip)
  - `bfa81c3` Ignore local run outputs (runs and src/wds)

**Action Required:** Review `git status` output before pushing. `src/Train_textalign.py` and `src/models_textalign.py` have significant modifications.

---

## 2. Large File Scan (>50MB)
**Critical Risks (Do NOT push to GitHub):**

| File Path | Size | Description |
|-----------|------|-------------|
| `./train_logs/s1_textalign_stage1_FINAL_BEST_32/last.pth` | **25G** | Best Stage 1 Model (Likely Release Candidate) |
| `./train_logs/s1_textalign_stage0_repair_80G/last.pth` | 23G | Stage 0 Checkpoint |
| `./src/coco_images_224_float16.hdf5` | **21G** | **HUGE DATASET** (Must Ignore) |
| `./unclip6_epoch0_step110000.ckpt` | **17G** | Stable Diffusion/UnCLIP Checkpoint |
| `./train_logs/final_multisubject_subj01/last.pth` | 9.6G | Multi-subject Checkpoint |
| `./convnext_xlarge_alpha0.75_fullckpt.pth` | 6.1G | Backbone Weights |
| `./evals/brain_tokens/...` | ~1.6G ea | Pre-computed embeddings |
| `./gnet_multisubject.pt` | 1.2G | GNet Weights |
| `./.cache/huggingface/...` | 1.2G | Cached HF Model |
| `./.git/objects/pack/...` | 1.2G | **Git Repo History is Large** |

**Summary of High-Risk Directories:**
- `train_logs/`: Contains multiple 5GB-25GB checkpoints.
- `evals/`: Contains GB-sized token files.
- `src/`: Contains `coco_images_224_float16.hdf5` (Wrong location for data?).

---

## 3. Key Artifacts & Weights
**Training Checkpoints (`train_logs/`):**
- **Top Candidate:** `train_logs/s1_textalign_stage1_FINAL_BEST_32/last.pth` (25G)
- Alternative: `train_logs/s1_textalign_direct_stage1/last.pth` (8.4G)
- Alternative: `train_logs/final_multisubject_subj01/last.pth` (9.6G)

**Data Artifacts (`data/nsd_text`):**
- `train_coco_hardnegs.jsonl`
- `train_coco_text_clip.pt`
- `train_coco_captions.json`

---

## 4. Release Checklist

### A. To GitHub
*Recommend pushing only source code and configuration.*
- [ ] `src/` (EXCLUDING `.hdf5`, `.pth`, `.pt`)
- [ ] `tools/` (Scripts)
- [ ] `generative-models/` (Codebase)
- [ ] `environment_mindeye21.yml` / `requirements_mindeye21.txt`
- [ ] `README.md`
- [ ] `LICENSE`

### B. To Ignore (Add to .gitignore)
*Double check `tools/suggested_gitignore.txt`*
- [ ] `src/coco_images_224_float16.hdf5` (CRITICAL)
- [ ] `unclip6_epoch0_step110000.ckpt`
- [ ] `train_logs/`
- [ ] `evals/`
- [ ] `results/`
- [ ] `data/`
- [ ] `wds/`

### C. To HuggingFace
*Upload these separately:*
1.  **Model Weights:** `train_logs/s1_textalign_stage1_FINAL_BEST_32/last.pth`
2.  **Backbones:** `convnext_xlarge_alpha0.75_fullckpt.pth` (if license permits redistribution)
3.  **Config:** `protocol_config.json` (if applicable)

---

## 5. Next Steps
1.  Run `cat tools/suggested_gitignore.txt >> .gitignore`
2.  Move `src/coco_images_224_float16.hdf5` to `data/` or delete if redundant.
3.  Commit code changes.
