# Data preparation

ConceptAlign expects the MindEye2 WebDataset layout and NSD stimulus metadata.
Users must obtain NSD and COCO directly under their original licenses; this
repository and its Hugging Face repositories do not redistribute them.

The Final14 caption mapping command, with its server path replaced by
`NSD_ROOT`, is:

```bash
python tools/prepare_train_coco_captions_from_stiminfo.py \
  --subj 1 \
  --data_path "$NSD_ROOT" \
  --out data/nsd_text/train_coco_captions.json
```

Counterfactual generation requires an explicit API key and endpoint. Never
commit credentials:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com
python tools/gen_hard_neg_captions_from_json_v2.py --help
```

Encode the accepted captions using
`tools/encode_hard_neg_captions_clip.py`. C2 and C3 use distinct encoded
negative assets as named in their training configurations.
