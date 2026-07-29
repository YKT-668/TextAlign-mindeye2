# Installation

The frozen archive records Python 3.12.7 on Linux. For a maintainable setup,
create the curated environment:

```bash
conda env create -f environment/environment.yml
conda activate conceptalign
pip install -r environment/requirements.txt
```

`environment/conda-linux-64.txt` is the exact base Conda package list captured
with Final14. It is retained for provenance and is not the recommended
cross-platform installation path.

Install a PyTorch build compatible with the CUDA driver on the target host.
The Final14 command uses bf16 and `accelerate`; CPU checks do not require CUDA.
Optional reconstruction metrics require `evaluate`, `sentence-transformers`,
and the model-specific dependencies listed in the requirements file.

Set portable paths rather than editing code:

```bash
export CONCEPTALIGN_ROOT="$PWD"
export NSD_ROOT=/path/to/nsd
export COCO_ROOT=/path/to/coco
export CHECKPOINT_ROOT="$PWD/checkpoints"
export RESULTS_ROOT="$PWD/results"
export HF_HOME="$PWD/.cache/huggingface"
```
