# Third-party notices

ConceptAlign is derived from
[MindEye2](https://github.com/MedARC-AI/MindEyeV2). The repository-level
`LICENSE` is the original MindEye2 MIT license and remains unchanged.
MindEye2 code retained for ConceptAlign includes the brain network, data
loading, diffusion prior, reconstruction, and shared utilities.

`src/generative_models/` contains code derived from
[Stability AI generative-models](https://github.com/Stability-AI/generative-models).
Its code and model license files are preserved in that directory, including
`LICENSE-CODE` and `model_licenses/`.

Runtime dependencies include PyTorch, OpenCLIP, Hugging Face Transformers,
Accelerate, WebDataset, and scientific Python packages. Their own licenses
apply. Model weights are not included.

NSD and COCO are not redistributed. Users must obtain them under the original
dataset terms. This project does not claim ownership of MindEye2, Stability AI
code, NSD, COCO, or third-party model weights.
