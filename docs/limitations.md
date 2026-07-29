# Limitations

- NSD and COCO must be obtained independently under their original licenses.
- Full training and enhanced reconstruction require high-memory CUDA hardware.
- Results depend on fixed third-party encoders and checkpoint revisions.
- The retained implementation preserves historical `textalign` names in
  modules and checkpoints for compatibility.
- The Final14 internal document's Stage0 label conflicts with its stage
  environment variable; the public configs expose this rather than silently
  rewriting the archived command.
- Participant-facing human-study pages and raw annotations are intentionally
  excluded. Only aggregate, non-identifying assets are mapped.
- `ours_ss2` is a diagnostic non-paper asset and is not published here.
