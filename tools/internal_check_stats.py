import json
import torch
import os
from pathlib import Path

def step1():
    print("--- Step 1 Output ---")
    p = Path("data/nsd_text/train_coco_captions.json")
    if p.exists():
        obj = json.loads(p.read_text())
        ids = obj.get("image_ids", [])
        caps = obj.get("captions", [])
        print("== train_coco_captions.json ==")
        print("len(image_ids) =", len(ids))
        print("len(captions)  =", len(caps))
        print("unique image_ids =", len(set(ids)))
    else:
        print(f"File not found: {p}")

def step2_2():
    print("\n--- Step 2.2 Output ---")
    path = "data/nsd_text/train_coco_captions_hard_negs_clip.pt"
    if os.path.exists(path):
        try:
            obj = torch.load(path, map_location="cpu")
            print("== hard_negs_clip.pt ==")
            print("type:", type(obj))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if hasattr(v, "shape"):
                        print(f"{k}: {tuple(v.shape)}")
            elif hasattr(obj, "shape"):
                print("tensor shape:", tuple(obj.shape))
            else:
                print("no tensor with .shape found")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"File not found: {path}")

def step3():
    print("\n--- Step 3 Output ---")
    path = "data/nsd_text/train_coco_text_clip.pt"
    if os.path.exists(path):
        try:
            obj = torch.load(path, map_location="cpu")
            print("== train_coco_text_clip.pt ==")
            print("type:", type(obj))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if hasattr(v, "shape"):
                        print(f"{k}: {tuple(v.shape)}")
            elif hasattr(obj, "shape"):
                print("tensor shape:", tuple(obj.shape))
            else:
                print("no tensor with .shape found")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"File not found: {path}")

if __name__ == "__main__":
    step1()
    step2_2()
    step3()
