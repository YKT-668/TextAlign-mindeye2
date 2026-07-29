# ConceptAlign

ConceptAlign aligns brain-derived visual representations with
stimulus-preserving counterfactual language for more semantically faithful
fMRI-to-image reconstruction.

> Paper status: manuscript under review. Paper and project-page links will be
> added after publication.

ConceptAlign, internally developed under the historical code name TextAlign,
is built upon the MindEye2 reconstruction framework.

## Method overview

The model adds a text-alignment projector to MindEye2. Predicted brain tokens
are aligned to a positive caption using batchwise InfoNCE and are separated
from near-miss counterfactual captions using a margin loss. The public tree
retains the training loop, projector, positive/negative construction tools,
checkpoint loading, reconstruction inference, and evaluation programs used by
the Final14 archive.

## Contributions

- Counterfactual semantic alignment for brain-derived visual tokens.
- Object, attribute, and relation near-miss negatives.
- Matched positive-only, random-negative, and CLIP-nearest ablations.
- Retrieval, reconstruction, 2AFC, CCD/CCD-H, RSA, IS-RSA, bootstrap, margin,
  low-data, and cross-subject evaluation paths.

## Repository structure

```text
configs/      Portable training, inference, evaluation, and artifact mappings
docs/         Installation, data, reproduction, evaluation, and result guides
environment/  Human-readable and exact Linux environment specifications
results/      Small machine-readable paper summaries and catalog
scripts/      Download and figure entry points
src/          ConceptAlign, retained MindEye2, and reconstruction code
tests/        CPU-only public release checks
tools/        Data preparation, evaluation, checksums, and catalog queries
```

## Installation

Python 3.10–3.12 is supported; the frozen Final14 environment used Python
3.12.7. A CUDA-capable Linux host is required for full training and
reconstruction. The CPU smoke tests do not require data or a GPU.

```bash
conda env create -f environment/environment.yml
conda activate conceptalign
export CONCEPTALIGN_ROOT="$PWD"
export NSD_ROOT=/path/to/nsd
export COCO_ROOT=/path/to/coco
export CHECKPOINT_ROOT="$PWD/checkpoints"
export RESULTS_ROOT="$PWD/results"
```

See [installation](docs/installation.md) for CUDA/PyTorch notes.

## Data preparation and licensing

Apply separately for the [Natural Scenes Dataset](https://naturalscenesdataset.org/)
and obtain [COCO](https://cocodataset.org/) under their original terms.
Neither dataset is redistributed here. Prepare caption alignment using:

```bash
python tools/prepare_train_coco_captions_from_stiminfo.py \
  --subj 1 --data_path "$NSD_ROOT" \
  --out data/nsd_text/train_coco_captions.json
```

See [data preparation](docs/data_preparation.md).

## Hugging Face artifacts

GitHub contains code, configuration, documentation, and small result tables.
Checkpoints are in
[`ykt668/textalign-mindeye2-model`](https://huggingface.co/ykt668/textalign-mindeye2-model);
intermediate outputs and supplemental results are in
[`ykt668/textalign-mindeye2-data`](https://huggingface.co/datasets/ykt668/textalign-mindeye2-data).

| Resource | Frozen revision |
|---|---|
| HF Model `main` | `ceb32860e6a9e1c775e30da8f8e884e0f1926795` |
| HF Model `master` | `a127f9295fd5656ac63ae436f07e61b80bf4efce` |
| HF Dataset | `2c612c5c2ca3c5344854edbd7028769cb254ab5a` |

Preview a pinned download without transferring data:

```bash
python scripts/download/download_artifacts.py ours_s1
```

## Quick start

Render a command before launching it:

```bash
python tools/config_command.py configs/training/ours.yaml
python tools/config_command.py configs/inference/final14.yaml
python tools/config_command.py configs/evaluation/main.yaml
```

Append `--execute` only after setting the documented paths and downloading the
required artifacts.

## Training

The canonical Stage1 values and command come from `src/readme3.md` at the
Final14 archive and are encoded in `configs/training/ours.yaml`. The public
configs cover shared Stage0, Ours, C1 positive-only, C2 random-negative, C3
CLIP-nearest, low-data, and cross-subject training.

```bash
python tools/config_command.py configs/training/shared_stage0.yaml --execute
python tools/config_command.py configs/training/ours.yaml --execute
```

## Inference

`src/recon_inference_run.py` exports decoded brain features, prior outputs,
blurry and enhanced reconstructions, predicted captions, and matched Final14
artifacts. The portable form of the archived command is:

```bash
python tools/config_command.py configs/inference/final14.yaml --execute
```

## Evaluation

Reconstruction metrics are provided by `src/run_debug.py` and
`tools/eval_recons.py`. Semantic evaluation entry points live under `tools/`
and cover retrieval, 2AFC, CCD/CCD-H, RSA, IS-RSA, bootstrap summaries,
margin ablations, semantic breakdown, and cross-subject checks. See
[evaluation](docs/evaluation.md) for exact invocations and the mapping for
Cross-LLM, Human-written, Human Audit, and Figure 4 assets.

## C1/C2/C3 ablations

```bash
python tools/config_command.py configs/training/c1_positive_only.yaml
python tools/config_command.py configs/training/c2_random_negative.yaml
python tools/config_command.py configs/training/c3_clip_nearest.yaml
```

All three use the shared Stage0 checkpoint and otherwise preserve the Final14
training settings.

## Existing results and catalog

Small result tables are in `results/tables/`. Query the artifact-oriented
catalog with:

```bash
python tools/results_catalog.py
python tools/results_catalog.py --experiment ours_s1
```

Full inference tensors, human experiments, supplemental experiments, and
Figure 4 inputs are pinned in the HF Dataset revision above. See
[results](docs/results.md) and [artifact mapping](docs/artifact_mapping.md).

## Reproducing paper tables

Download `final14_results`, then use the evaluation tools and figure programs
described in [reproduction](docs/reproduction.md). The checked-in tables are
small reference summaries, not substitutes for recomputation.

## Hardware

Full MindEye2/ConceptAlign training and enhanced reconstruction require a
CUDA GPU with substantial memory; Final14 used single-process bf16 for the
documented Stage1 command. Evaluation-only memory requirements depend on the
embedding representation. Smoke tests run on CPU.

## Limitations

The code does not redistribute NSD, COCO, checkpoints, or participant data.
Reproduction depends on licensed datasets, fixed third-party models, and
high-memory GPU hardware. See [limitations](docs/limitations.md).

## License and attribution

The repository retains its original license in [LICENSE](LICENSE).
ConceptAlign builds on [MindEye2](https://github.com/MedARC-AI/MindEyeV2) and
bundles portions of Stability AI's generative-models code. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the preserved license
files in `src/generative_models/`.

## Citation

Use [CITATION.cff](CITATION.cff). Replace the paper placeholder after
publication.

## Contact

Project contact: **to be added by the authors before public release**.
