#!/usr/bin/env python
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import random

# --- 1. 定义与 DeepSeek API 交互的核心函数 ---
def get_structured_prompt_from_deepseek(topk_texts, client, max_retries=2):
    """
    调用 DeepSeek API，将一组Top-K文本转换为一个结构化的prompt。
    添加了重试机制和更短的超时。
    """
    # 精心设计的"系统指令"，告诉DeepSeek它的角色和任务
    system_prompt = """
You are an expert AI art prompt engineer. Your task is to synthesize a set of descriptive sentences about a scene into a high-quality, structured prompt for a text-to-image model like Stable Diffusion XL.

The user will provide a list of sentences under "Top-K Descriptions".

Your response MUST be a single, valid JSON object, with no other text before or after it.
The JSON object must have exactly two keys:
1. "positive": A concise, vivid, and coherent description of the main scene. Focus on key objects, their actions, and the overall atmosphere. Do NOT use superlative or generic art terms like '4k', '8k', 'highly detailed', 'cinematic', 'masterpiece'.
2. "negative": A standard list of negative keywords to avoid common image generation artifacts.

Example Input:
[
    "A black and white photo of a man in a suit and tie.",
    "A man in a suit and tie is standing in front of a building.",
    "A man in a suit and tie is looking at the camera.",
    "A black and white photo of a man in a suit.",
    "A man in a suit and tie is standing in front of a building with a clock on it."
]

Example Output:
{
  "positive": "A man in a black suit and tie standing in front of a building with a large clock, looking at the camera. Black and white photo.",
  "negative": "blurry, low quality, artifacts, extra limbs, text, watermark, copyright, deformed, mutated, ugly"
}
"""
    
    # 将Top-K文本列表格式化为用户输入
    user_content = "Please synthesize the following descriptions into a structured prompt:\n\nTop-K Descriptions:\n" + json.dumps(topk_texts, indent=2)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
                max_tokens=300,  # 减少token数量以提速
                temperature=0.3,  # 进一步降低温度提速
                timeout=15,  # 设置较短的超时时间
                response_format={"type": "json_object"}, # 请求JSON输出
            )
            
            # 解析返回的JSON字符串
            message_content = response.choices[0].message.content
            structured_prompt = json.loads(message_content)
            
            # 验证返回的JSON是否符合我们的期望
            if "positive" in structured_prompt and "negative" in structured_prompt:
                return structured_prompt
            else:
                if attempt == max_retries:
                    print(f"⚠️ Warning: DeepSeek response missing required keys. Got: {message_content}")
                return None

        except Exception as e:
            if attempt == max_retries:
                print(f"❌ Error calling DeepSeek API after {max_retries + 1} attempts: {e}")
                return None
            # 随机延迟重试，避免同时重试
            time.sleep(random.uniform(0.1, 0.5))
    
    return None

def process_batch_worker(batch_data, client, results_queue, progress_queue):
    """
    工作线程函数，处理一批API请求
    """
    batch_results = []
    for rec in batch_data:
        topk_texts = rec.get("topk", [])
        if not topk_texts:
            continue

        # 调用 DeepSeek API
        structured_result = get_structured_prompt_from_deepseek(topk_texts, client)
        
        if structured_result:
            # 将返回的结果与原始ID结合
            structured_result['id'] = rec.get("id")
            batch_results.append(structured_result)
        
        # 更新进度
        progress_queue.put(1)
        
        # 减少延迟，仅在必要时暂停
        time.sleep(0.05)  # 极短暂停，避免过载
    
    results_queue.put(batch_results)

# --- 2. 主函数 ---
def main():
    ap = argparse.ArgumentParser(description="Generate structured prompts from Top-K texts using DeepSeek LLM.")
    ap.add_argument("--topk_jsonl", required=True, help="Path to the input JSONL file from the retrieval step.")
    ap.add_argument("--out_json", required=True, help="Path to the output JSON file for the final prompts.")
    # API Key 可以通过环境变量或直接参数传入
    ap.add_argument("--api_key", default=os.environ.get('DEEPSEEK_API_KEY'), help="DeepSeek API Key.")
    ap.add_argument("--max_workers", type=int, default=8, help="Number of concurrent workers for API calls.")
    ap.add_argument("--batch_size", type=int, default=50, help="Batch size for processing records.")
    args = ap.parse_args()

    if not args.api_key:
        raise ValueError("DeepSeek API Key not found. Please set the DEEPSEEK_API_KEY environment variable or provide it via --api_key.")

    print(f"🚀 Starting with {args.max_workers} workers, batch size {args.batch_size}")

    # 读取输入的 JSONL 文件
    recs = []
    with open(args.topk_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    
    print(f"📊 Processing {len(recs)} records")
    
    # 分批处理数据
    batches = [recs[i:i + args.batch_size] for i in range(0, len(recs), args.batch_size)]
    
    final_prompts = []
    results_queue = Queue()
    progress_queue = Queue()
    
    # 创建进度条
    total_records = len(recs)
    pbar = tqdm(total=total_records, desc="Generating Prompts with DeepSeek")
    
    def progress_updater():
        """更新进度条的线程"""
        processed = 0
        while processed < total_records:
            try:
                progress_queue.get(timeout=1)
                processed += 1
                pbar.update(1)
            except:
                continue
        pbar.close()
    
    # 启动进度更新线程
    progress_thread = threading.Thread(target=progress_updater, daemon=True)
    progress_thread.start()
    
    # 使用线程池并发处理批次
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 为每个线程创建独立的客户端实例
        clients = [OpenAI(
            api_key=args.api_key,
            base_url="https://api.deepseek.com"
        ) for _ in range(args.max_workers)]
        
        # 提交批次任务
        futures = []
        for i, batch in enumerate(batches):
            client = clients[i % len(clients)]  # 循环使用客户端
            future = executor.submit(process_batch_worker, batch, client, results_queue, progress_queue)
            futures.append(future)
        
        # 收集所有结果
        completed_batches = 0
        for future in as_completed(futures):
            try:
                future.result()  # 确保任务完成
                completed_batches += 1
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
    
    # 收集所有结果
    while not results_queue.empty():
        batch_results = results_queue.get()
        final_prompts.extend(batch_results)
    
    # 等待进度更新线程完成
    progress_thread.join(timeout=2)
    
    print(f"\n📈 Performance: Processed {len(recs)} records, generated {len(final_prompts)} prompts")
    print(f"📊 Success rate: {len(final_prompts)}/{len(recs)} ({len(final_prompts)/len(recs)*100:.1f}%)")

    # 保存最终结果
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(final_prompts, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ WROTE (AI Generated): {args.out_json} ({len(final_prompts)} prompts)")

if __name__ == "__main__":
    main()
