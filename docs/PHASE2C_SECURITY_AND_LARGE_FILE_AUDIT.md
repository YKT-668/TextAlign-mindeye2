# Phase 2C Security and Large File Audit

**Date:** 2026-06-11
**Repo:** TextAlign-mindeye2

## 1. Large File Scan (>50MB)

| Check | Result |
|-------|--------|
| Files >50MB in repo | **Zero** — none found |
| Large binaries (pt/pth/ckpt/safetensors) | None in tracked files |

**Verdict:** No large files in staging risk. ✅

## 2. Sensitive Token Scan

Scanned for patterns: `hf_`, `HUGGINGFACE`, `token`, `api_key`, `secret`, `WANDB_API_KEY`

| Pattern | Matches | Nature |
|---------|---------|--------|
| `hf_` | ~30 matches | All are environment variable names (`HF_ENDPOINT`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`), library imports (`huggingface_hub`, `hf_hub_download`), or path references (`hf_cache/`, `hf_textalign_mindeye2_model/`) |
| `token` | ~5 matches | Only references to `HuggingFace token` in documentation — no actual tokens |
| `api_key` | ~5 matches | `DEEPSEEK_API_KEY` used as env var check in `run_official_hf_baselines.py` — not hardcoded |
| `WANDB_API_KEY` | 0 matches | Not present anywhere |

## 3. Verdict

**Safe to proceed.** No sensitive keys, tokens, or API secrets are hardcoded in any file within the repo. All references are:
- Environment variable names (not values)
- Library import references
- Documentation mentions of required credentials (without actual values)

## 4. Files Recommended for `.gitignore`

| Pattern | Reason |
|---------|--------|
| `hf_cache/` | HuggingFace cache — can contain token files |
| `checkpoints/` | Large binary files |
| `features/` | Large binary files |
| `data/` | NSD/COCO raw data |
| `outputs/` | Local debug outputs |
| `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` | Model weight binaries |
| `*.zip`, `*.tar` | Archive files |
| `*.log` | Training logs |
| `.env` | Environment secrets |
