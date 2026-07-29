# Results

`results/tables/` contains compact machine-readable summaries retained from
the Final14 source. Query the public catalog:

```bash
python tools/results_catalog.py
python tools/results_catalog.py --experiment ours_s1
```

The HF Dataset repository contains the full Final14 inference outputs,
evaluation tensors, human-written and cross-LLM experiments, Human Audit
aggregate, margin/low-data supplements, Figure 4 inputs, and paper materials:

```text
ykt668/textalign-mindeye2-data
revision 2c612c5c2ca3c5344854edbd7028769cb254ab5a
```

Checked-in tables are convenient reference summaries. Recomputed values should
be written beneath `RESULTS_ROOT` and compared only after ID and protocol
alignment.
