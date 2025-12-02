#!/usr/bin/env python
#接收一堆零散的“灵感”（Top-K文本），然后运用你为它设定的规则（拼接、合并、清理），最终“精炼”并“锻造”出可以直接喂给SDXL图像生成模型的、高质量的、结构化的“最终指令”（prompt_llm.json）。
import os, json, argparse
import re # 引入正则表达式库，用于清理文本

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", required=True, help="Path to the input JSONL file from retrieve_topk.py.")
    ap.add_argument("--out",  required=True, help="Path to the output JSON file for prompts.")
    ap.add_argument("--neg",  default="blurry, low quality, artifacts, extra limbs, text, watermark, copyright")
    # 从 style 的默认值中移除我们打算清理掉的词，让逻辑更清晰
    ap.add_argument("--style", default="photographic, realistic")
    args = ap.parse_args()

    # --- 1. 定义清理函数 (将清理逻辑内聚到脚本内部) ---
    def clean_and_clamp_prompt(text_prompt, style_prompt, neg_prompt):
        """
        合并、清理并压缩prompt，使其尽可能符合CLIP的77 token限制。
        """
        # 合并 positive 和 style
        full_positive = f"{text_prompt}, {style_prompt}"
        
        # 定义要移除的冗余词列表
        # 你可以根据需要随时在这里添加或修改
        redundant_words = [
            r"\b4\s*k\b",
            r"\b8\s*k\b",
            r"\bultra[-\s]*detailed\b",
            r"\bhighly[-\s]*detailed\b",
            r"\bcinematic\b",
            r"\bhyper[-\s]*realistic\b"
        ]
        
        # 对 positive 和 negative prompts 应用清理规则
        for word_pattern in redundant_words:
            full_positive = re.sub(word_pattern, "", full_positive, flags=re.I)
            neg_prompt = re.sub(word_pattern, "", neg_prompt, flags=re.I)
            
        # 压缩多余的空格和逗号
        def compress_text(s):
            s = re.sub(r"\s+,", ",", s)
            s = re.sub(r",\s+", ", ", s)
            s = re.sub(r"\s+", " ", s)
            return s.strip(" ,")

        return compress_text(full_positive), compress_text(neg_prompt)

    # --- 2. 读取 JSONL 文件 (逻辑保持不变) ---
    recs = []
    with open(args.topk, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    
    # --- 3. 生成最终的、已清理的 prompts ---
    final_prompts = []
    for rec in recs:
        cands = rec.get("topk", [])
        if not cands:
            continue

        # 拼接原始的 positive prompt
        raw_positive_text = f"Main scene: {cands[0]} || Alternatives: " + " | ".join(cands[1:])
        
        # --- 在这里调用我们新的清理和合并函数 ---
        final_positive, final_negative = clean_and_clamp_prompt(
            raw_positive_text, 
            args.style, 
            args.neg
        )
        
        final_prompts.append({
            "id": rec.get("id"), 
            "positive": final_positive,
            "negative": final_negative,
            # style 字段现在是多余的，我们不再输出它，让prompt文件更干净
        })
        
    # --- 4. 保存结果 (逻辑保持不变) ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(final_prompts, open(args.out,"w"), ensure_ascii=False, indent=2)
    print(f"WROTE (cleaned and merged): {args.out} ({len(final_prompts)} prompts)")

if __name__ == "__main__":
    main()
