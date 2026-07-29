# Reproduction

The command authority for the final S1 training, inference, and reconstruction
evaluation is `src/readme3.md` at
`archive/final14@231850a314ab85fc99e0238b1eee7ccd56c01155`. Its portable
Stage1 and inference forms are encoded in:

- `configs/training/ours.yaml`
- `configs/inference/final14.yaml`
- `configs/evaluation/main.yaml`

Render each command before execution:

```bash
python tools/config_command.py configs/training/ours.yaml
python tools/config_command.py configs/inference/final14.yaml
python tools/config_command.py configs/evaluation/main.yaml
```

The archived document labels one repair command “Stage0” while setting
`MINDEYE_TEXTALIGN_STAGE=1`. The public `shared_stage0.yaml` follows the
trainer's explicit Stage0 semantics (`0`, text head only); `ours.yaml`
preserves the archived Stage1 values exactly. This discrepancy is recorded for
author review and does not silently change the training implementation.

For the paper ablations, use the C1/C2/C3 configs. For low-data and
cross-subject runs, use their corresponding configs and preserve the same
checkpoint, seed, representation, and evaluation subset across comparisons.
These files are marked `author_confirmation_required` because their exact
commands are not recorded in `readme3`; `tools/config_command.py` refuses to
execute them unless an author explicitly passes `--allow-unverified`.
