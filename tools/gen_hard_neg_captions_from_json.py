#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 train_coco_captions.json 读取正样本 caption，
调用 DeepSeek API 为每个 image_id 生成一个 hard negative caption。

输入:
    data/nsd_text/train_coco_captions.json
    {
        "image_ids": [...],
        "captions":  [...]
    }

输出:
    data/nsd_text/train_coco_captions_hard_negs.jsonl

    每行一条 JSON:
    {
        "image_id":   int,
        "pos_caption": str,
        "neg_caption": str
    }

支持断点续跑：若输出文件已存在，会跳过已经处理过的 image_id。
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI


# ============== 路径配置 ==============

PROJ_ROOT = Path(__file__).resolve().parents[1]
IN_JSON = PROJ_ROOT / "data/nsd_text/train_coco_captions.json"
OUT_JSONL = PROJ_ROOT / "data/nsd_text/train_coco_captions_hard_negs.jsonl"


# ============== DeepSeek 客户端 ==============

def build_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置，请先 export DEEPSEEK_API_KEY=...")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    return client


SYSTEM_PROMPT = (
    "You are an expert at generating hard negative captions for image-text contrastive learning.\n"
    "The user will provide an original English caption that correctly describes an image.\n"
    "Your task: generate ONE short English sentence that is semantically close to the original "
    "but clearly WRONG as a description of that image.\n"
    "Requirements:\n"
    "1. It must be plausible and natural, not random nonsense.\n"
    "2. It must contradict at least one key attribute: object category, number of objects, color, action, or scene.\n"
    "3. It should stay in the same general topic (e.g., still about sports if the original is sports).\n"
    "4. Output ONLY the new caption text, without quotes or any explanations.\n"
)


def build_user_prompt(pos_caption: str) -> str:
    return (
        "Original (positive) caption:\n"
        f"\"{pos_caption}\"\n\n"
        "Now generate ONE hard negative caption that is similar in style and length, but factually incorrect "
        "about the same image. Remember: output only the new caption."
    )


def gen_one_neg_caption(client: OpenAI, pos_caption: str, max_retries: int = 3) -> str:
    """调用 DeepSeek，为单个正样本 caption 生成一个 hard negative caption。"""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(pos_caption)},
                ],
                temperature=0.8,
                max_tokens=64,
                stream=False,
            )
            text = resp.choices[0].message.content.strip()

            # 简单清洗：去掉首尾引号
            if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) > 2:
                text = text[1:-1].strip()

            # 防止输出空的
            if not text:
                raise ValueError("Empty neg caption")

            return text
        except Exception as e:
            print(f"[WARN] 调用 DeepSeek 失败 (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(2.0 * attempt)  # 简单退避


# ============== 读取正样本 JSON ==============

def load_pos_json(path: Path) -> Dict[str, List[Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到输入文件: {path}")

    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"期望 JSON 为 dict, 实际为: {type(obj)}")

    if "image_ids" not in obj or "captions" not in obj:
        raise ValueError("JSON 中必须包含键 'image_ids' 和 'captions'")

    image_ids = obj["image_ids"]
    captions = obj["captions"]
    if len(image_ids) != len(captions):
        raise ValueError(f"image_ids 数量({len(image_ids)}) 与 captions 数量({len(captions)}) 不一致")

    return {"image_ids": image_ids, "captions": captions}


# ============== 已处理记录 (断点续跑) ==============

def load_done_ids(out_path: Path) -> set:
    done = set()
    if out_path.is_file():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    iid = obj.get("image_id", None)
                    if iid is not None:
                        done.add(int(iid))
                except Exception:
                    continue
    return done


# ============== 主流程 ==============

def main():
    print(f"[INFO] 读取正样本: {IN_JSON}")
    data = load_pos_json(IN_JSON)
    image_ids: List[int] = data["image_ids"]
    captions: List[str] = data["captions"]

    print(f"[INFO] 共 {len(image_ids)} 条 (image_id, caption)")

    done_ids = load_done_ids(OUT_JSONL)
    if done_ids:
        print(f"[INFO] 检测到已有输出 {OUT_JSONL}, 已完成 {len(done_ids)} 条, 将跳过这些 image_id")

    client = build_client()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("a", encoding="utf-8") as fout:
        for idx, (iid, cap) in enumerate(zip(image_ids, captions), start=1):
            if int(iid) in done_ids:
                continue

            pos_cap = cap.strip()
            if not pos_cap:
                print(f"[WARN] image_id={iid} 的 caption 为空, 跳过")
                continue

            print(f"[{idx}/{len(image_ids)}] image_id={iid}")
            print(f"  POS: {pos_cap}")

            try:
                neg_cap = gen_one_neg_caption(client, pos_cap)
            except Exception as e:
                print(f"[ERROR] 生成 neg caption 失败, image_id={iid}, 跳过. Error: {e}")
                continue

            print(f"  NEG: {neg_cap}\n")

            rec = {
                "image_id": int(iid),
                "pos_caption": pos_cap,
                "neg_caption": neg_cap,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            # 小小的 sleep，避免 QPS 过高
            time.sleep(0.3)


if __name__ == "__main__":
    main()
