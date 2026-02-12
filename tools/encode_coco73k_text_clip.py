#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把 MindEye2 的 evals/all_captions.pt 编码成 CLIP 文本向量，
得到一个和 73K COCO IDX 对齐的 teacher 特征表：
    text_feats[cocoidx] ∈ R^d_text
"""

import os
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel


def load_captions(evals_root: str = "evals"):
    """
    读取 evals/all_captions.pt，将其转成一个长度为 N 的字符串列表。
    尽量兼容 list / dict / numpy.ndarray 等多种结构。
    """
    path = os.path.join(evals_root, "all_captions.pt")
    print(f"[INFO] Loading captions from {path}")

    # 关键：显式关闭 weights_only 限制
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[INFO] type(all_captions) = {type(obj)}")

    # ---------- 情况 A：list / tuple ----------
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError("all_captions.pt is empty (list/tuple len=0).")

        first = obj[0]
        print(f"[INFO] list/tuple first element type = {type(first)}")

        if isinstance(first, str):
            # 直接是字符串列表
            return list(obj)

        if isinstance(first, (list, tuple)):
            # 每个是 [cap1, cap2, ...]，取第一条
            return [caps[0] for caps in obj]

        raise RuntimeError(
            f"Unsupported list element type: {type(first)}; "
            f"please send me this type so我能改解析逻辑。"
        )

    # ---------- 情况 B：dict ----------
    if isinstance(obj, dict):
        print(f"[INFO] dict keys = {list(obj.keys())}")
        if "captions" in obj:
            caps = obj["captions"]
        elif "all_captions" in obj:
            caps = obj["all_captions"]
        else:
            raise RuntimeError(
                f"dict keys = {list(obj.keys())}, "
                f"找不到 'captions' 或 'all_captions'，需要你把结构发给我。"
            )

        # 递归用上面的逻辑再解析一次
        if isinstance(caps, (list, tuple)):
            if len(caps) == 0:
                raise RuntimeError("captions list is empty.")
            if isinstance(caps[0], str):
                return list(caps)
            if isinstance(caps[0], (list, tuple)):
                return [c[0] for c in caps]
            raise RuntimeError(
                f"Unsupported captions[0] type in dict: {type(caps[0])}"
            )

        if isinstance(caps, np.ndarray):
            obj = caps  # 交给后面的 numpy 分支处理
        else:
            raise RuntimeError(
                f"Unsupported 'captions' type in dict: {type(caps)}"
            )

    # ---------- 情况 C：numpy.ndarray ----------
    if isinstance(obj, np.ndarray):
        print(f"[INFO] numpy.ndarray shape = {obj.shape}, dtype = {obj.dtype}")

        # 情况 C1: 一维数组，每个元素就是一个字符串
        if obj.ndim == 1 and (
            np.issubdtype(obj.dtype, np.str_) or np.issubdtype(obj.dtype, np.unicode_)
        ):
            return [str(x) for x in obj]

        # 情况 C2: 一维数组，dtype=object
        if obj.ndim == 1 and obj.dtype == object:
            first = obj[0]
            print(f"[INFO] ndarray[0] type = {type(first)}")

            if isinstance(first, str):
                # 直接是 object-string
                return [str(x) for x in obj]

            if isinstance(first, (list, tuple, np.ndarray)):
                # 每个元素是一个 [cap1, cap2, ...]，取第一条
                def pick_first(x):
                    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
                        return str(x[0])
                    return str(x)

                return [pick_first(x) for x in obj]

            raise RuntimeError(
                f"Unsupported ndarray object element type: {type(first)}; "
                f"需要你把具体结构发我看一下。"
            )

        # 情况 C3: 二维数组，可能是 (N, K) 的字符串
        if obj.ndim == 2 and (
            np.issubdtype(obj.dtype, np.str_) or np.issubdtype(obj.dtype, np.unicode_)
        ):
            # 对每一行做 join 或取第一列，这里简单取第一列
            return [str(x[0]) for x in obj]

        raise RuntimeError(
            f"暂不支持的 numpy.ndarray 结构: shape={obj.shape}, dtype={obj.dtype}；"
            f"请在 Python 里手动打印几行 obj[0], obj[1] 发给我。"
        )

    # ---------- 其他情况 ----------
    raise RuntimeError(
        f"暂时无法自动解析 all_captions.pt 的结构，"
        f"type(obj) = {type(obj)}。"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    captions = load_captions("evals")
    n = len(captions)
    print(f"[INFO] Loaded {n} captions from evals/all_captions.pt")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval().requires_grad_(False)

    all_feats = []
    bs = 64

    for i in range(0, n, bs):
        batch_caps = captions[i:i + bs]
        inputs = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)  # [B, d_text]
        all_feats.append(feats.cpu())

        if (i // bs) % 100 == 0:
            print(f"[INFO] Encoded {min(i + bs, n)}/{n} captions...")

    all_feats = torch.cat(all_feats, dim=0)  # [N, d_text]
    print("[INFO] Final feats shape:", all_feats.shape)

    out_dir = "data/nsd_text"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coco73k_text_clip.pt")

    torch.save(
        {
            "text_feats": all_feats,   # [N, d_text]，N 应该是 73K
            "model_name": "openai/clip-vit-large-patch14",
        },
        out_path,
    )
    print(f"[OK] Saved CLIP text features to {out_path}")


if __name__ == "__main__":
    main()
