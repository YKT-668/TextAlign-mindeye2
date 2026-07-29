# Artifact mapping

## Frozen anchors

- GitHub archive source: `archive/final14`
- GitHub source commit: `231850a314ab85fc99e0238b1eee7ccd56c01155`
- HF Model main: `ceb32860e6a9e1c775e30da8f8e884e0f1926795`
- HF Model master: `a127f9295fd5656ac63ae436f07e61b80bf4efce`
- HF Dataset: `2c612c5c2ca3c5344854edbd7028769cb254ab5a`

## Checkpoints

| Logical artifact | Repository branch/revision | Path |
|---|---|---|
| C1 | model main / pinned main revision | `checkpoints/final14_ablation/positive_only/last.pth` |
| C2 | model main / pinned main revision | `checkpoints/final14_ablation/random_negative/last.pth` |
| C3 | model main / pinned main revision | `checkpoints/final14_ablation/clip_nearest/last.pth` |
| Shared Stage0 | model main / pinned main revision | `checkpoints/final14_ablation/shared_stage0/last.pth` |
| Ours S1 | model master / pinned master revision | `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth` |
| Ours S2 | model main / `ceb32860e6a9e1c775e30da8f8e884e0f1926795` | `snapshots/2026-04-13/train_logs/s2_textalign_stage1_FINAL_BEST_32/last.pth` |
| Ours S5 | model main / `ceb32860e6a9e1c775e30da8f8e884e0f1926795` | `snapshots/2026-04-13/train_logs/s5_textalign_stage1_FINAL_BEST_32/last.pth` |
| Ours S7 | model main / `ceb32860e6a9e1c775e30da8f8e884e0f1926795` | `snapshots/2026-04-13/train_logs/s7_textalign_stage1_FINAL_BEST_32/last.pth` |

## Results

The results catalog, inference outputs, supplemental experiments, Cross-LLM,
Human-written, Human Audit aggregate, and Figure 4 assets are under
`final14_archive/` in the pinned HF Dataset revision. The `ours_ss2`
checkpoint is a diagnostic non-paper asset and is not part of this release.

`configs/artifacts.json` provides the machine-readable subset used by the
download helper.
