# Evaluation

Final14 separates reconstruction quality from semantic alignment.

| Protocol | Entry point |
|---|---|
| Reconstruction metrics and retrieval | `src/run_debug.py`, `tools/eval_recons.py` |
| Latent retrieval | `tools/eval_textalign_latent_plus.py` |
| 2AFC | `tools/eval_twoafc_embed.py` |
| CCD / CCD-H | `tools/eval_ccd_embed.py` |
| CCD margin and difficulty ablation | `tools/run_ccd_ablation.py` |
| Semantic category breakdown | `tools/make_ccd_by_type.py` |
| RSA | `tools/eval_rsa_embed.py` |
| IS-RSA | `tools/eval_isrsa_shared982.py` |
| Bootstrap and paired CI | evaluation outputs plus `results/tables/ci_bootstrap.csv` |
| Cross-subject | `tools/audit_cross_subject.py` |

The Cross-LLM, Human-written, Human Audit aggregate, supplemental low-data
outputs, and Figure 4 inputs are stored in the pinned HF Dataset Final14 tree.
They are result assets rather than reusable public participant-facing
software. Use the catalog and artifact mapping to locate them.

The main CCD configuration can be rendered with:

```bash
python tools/config_command.py configs/evaluation/main.yaml
```

Keep IDs, negative pools, seeds, and bootstrap counts matched when comparing
methods. Do not mix shared1000 and shared982 without the recorded mask.
