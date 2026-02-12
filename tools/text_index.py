#赋能RAG: 创建好的 text_index_vith.pt 就是一个向量数据库。在下一步 (retrieve_topk.py) 中，我们就可以用一个查询向量（比如从脑信号转换来的向量）在这个数据库中进行“向量搜索”，瞬间找到语义上最匹配的几句话，从而为后续的LLM生成提供精准的上下文信息。所以，text_index.py 的本质工作就是：为你庞大的文本知识库建立一个基于AI语义理解的、可供快速检索的数字索引。
import torch, argparse, os, math, time
import open_clip

ap = argparse.ArgumentParser(description="Build a CLIP text index (ViT-H-14) from a captions .pt (list[str])")
ap.add_argument("--captions_pt", required=True, help="Input .pt file: list[str] or list[(str,...)]")
ap.add_argument("--out_pt", required=True, help="Output tensor file path")
ap.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding")
ap.add_argument("--device", default="cuda", help="Device to run encoding ('cuda' or 'cpu')")
ap.add_argument("--model", default="ViT-H-14", help="OpenCLIP model name")
ap.add_argument("--pretrained", default="laion2b_s32b_b79k", help="OpenCLIP pretrained tag")
ap.add_argument("--max_samples", type=int, default=0, help="Optionally limit number of captions (0 = no limit)")
ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N batches")
args = ap.parse_args()

start_time = time.time()
caps = torch.load(args.captions_pt, map_location="cpu", weights_only=False)
texts = [c[0] if isinstance(c, (list, tuple)) else c for c in caps]
if args.max_samples > 0:
    texts = texts[:args.max_samples]

print(f"[index] loading model {args.model} ({args.pretrained}) on {args.device}")
model, _, _ = open_clip.create_model_and_transforms(args.model, args.pretrained, device=args.device)
tokenizer = open_clip.get_tokenizer(args.model)

bs = args.batch_size
total = len(texts)
num_batches = math.ceil(total / bs)
print(f"[index] total captions={total}, batch_size={bs}, num_batches={num_batches}")

embs = []
last_log = time.time()
with torch.no_grad(), torch.cuda.amp.autocast(enabled=(args.device.startswith('cuda'))):
    for bi in range(num_batches):
        s = bi * bs
        e = min(s + bs, total)
        chunk = texts[s:e]
        tok = tokenizer(chunk)
        tok = tok.to(args.device)
        te = model.encode_text(tok)
        te = te / te.norm(dim=-1, keepdim=True)
        embs.append(te.float().cpu())
        if (bi + 1) % args.progress_every == 0 or (bi + 1) == num_batches:
            dt = time.time() - last_log
            pct = (bi + 1) / num_batches * 100
            print(f"  [progress] batch {bi+1}/{num_batches} ({pct:.1f}%), dt_since_last={dt:.2f}s")
            last_log = time.time()

out = torch.cat(embs, 0)
os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
torch.save(out, args.out_pt)
elapsed = time.time() - start_time
print(f"[done] saved: {args.out_pt} shape={tuple(out.shape)} elapsed={elapsed:.1f}s")

