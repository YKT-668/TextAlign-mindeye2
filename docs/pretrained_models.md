# Pretrained models

Use only pinned revisions for recovery.

| Artifact | Repository / revision | Path |
|---|---|---|
| Shared Stage0 | HF Model `ceb32860e6a9e1c775e30da8f8e884e0f1926795` | `checkpoints/final14_ablation/shared_stage0/last.pth` |
| C1 positive-only | same | `checkpoints/final14_ablation/positive_only/last.pth` |
| C2 random-negative | same | `checkpoints/final14_ablation/random_negative/last.pth` |
| C3 CLIP-nearest | same | `checkpoints/final14_ablation/clip_nearest/last.pth` |
| Ours S1 | HF Model `a127f9295fd5656ac63ae436f07e61b80bf4efce` | `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth` |

Ours S2/S5/S7 are stored under the April snapshot paths in the HF Model main
history and are described in `docs/artifact_mapping.md`. Download commands are
generated from `configs/artifacts.json`:

```bash
python scripts/download/download_artifacts.py c1
python scripts/download/download_artifacts.py ours_s1
```

Use `tools/checksum.py` to verify a supplied SHA-256 value.
