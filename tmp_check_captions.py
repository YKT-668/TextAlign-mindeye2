import json
from pathlib import Path
import os

p = Path("data/nsd_text/train_coco_captions.json")
if not p.exists():
    print(f"File not found: {p}")
else:
    obj = json.loads(p.read_text())
    assert isinstance(obj, dict), type(obj)
    image_ids = obj.get("image_ids", [])
    captions = obj.get("captions", [])
    print("len(image_ids) =", len(image_ids))
    print("len(captions)  =", len(captions))
    print("first 5 image_ids:", image_ids[:5])
    print("first 5 captions:", captions[:5])
    assert len(image_ids) == len(captions)
