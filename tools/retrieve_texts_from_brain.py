#!/usr/bin/env python
import os, json, torch, argparse
import open_clip
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(description="Retrieve top-K texts for brain vectors from a caption library using a specified CLIP model.")
    ap.add_argument("--brain_vec_pt", required=True, help="Path to the brain vectors .pt file.")
    ap.add_argument("--ids_json", required=True, help="Path to the JSON file containing sample IDs.")
    ap.add_argument("--captions_pt", required=True, help="Path to the all_captions.pt file.")
    ap.add_argument("--out_jsonl", required=True, help="Path to save the output .jsonl file.")
    ap.add_argument("--clip_model", type=str, default="ViT-bigG-14", help="Name of the CLIP model to use for encoding (e.g., ViT-bigG-14).")
    ap.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k", help="Name of the pretrained weights for the CLIP model.")
    ap.add_argument("--topk", type=int, default=8, help="Number of top texts to retrieve.")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for text encoding.")
    ap.add_argument("--device", default="cuda", help="Device to run on.")
    args = ap.parse_args()

    print("Loading data...")
    V = torch.load(args.brain_vec_pt, map_location="cpu").float()
    ids = json.load(open(args.ids_json))
    try:
        caps = torch.load(args.captions_pt, map_location="cpu", weights_only=False)
    except Exception:
        caps = torch.load(args.captions_pt, map_location="cpu")

    if isinstance(caps[0], (list, tuple)):
        caps = [c[0] for c in caps]

    print(f"Loaded {V.shape[0]} brain vectors (dim={V.shape[1]}) and {len(caps)} captions.")

    print(f"Loading CLIP model: {args.clip_model} ({args.clip_pretrained})")
    
    # 对于 ViT-bigG-14，使用 open_clip 并明确指定下载源
    if args.clip_model == "ViT-bigG-14":
        try:
            print("Loading ViT-bigG-14 via open_clip with proper pretrained weights...")
            # 确保使用正确的权重名称格式
            model = open_clip.create_model(args.clip_model, pretrained=args.clip_pretrained)
            model = model.to(args.device)
            model.eval()
            tok = open_clip.get_tokenizer(args.clip_model)
            print(f"✓ Successfully loaded {args.clip_model}")
        except Exception as e:
            print(f"Failed with open_clip: {e}")
            print("Attempting manual download from open_clip repository...")
            
            # 手动从 open_clip 的权重仓库加载
            import urllib.request
            import tempfile
            
            weights_url = "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin"
            print(f"Downloading weights from: {weights_url}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_file:
                urllib.request.urlretrieve(weights_url, tmp_file.name)
                
                # 创建模型架构
                model = open_clip.create_model(args.clip_model, pretrained=False)
                
                # 加载权重
                state_dict = torch.load(tmp_file.name, map_location="cpu")
                model.load_state_dict(state_dict)
                model = model.to(args.device)
                model.eval()
                
                print("✓ Successfully loaded from manual download")
            
            tok = open_clip.get_tokenizer(args.clip_model)
    else:
        # 其他模型使用标准方法
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model, 
            pretrained=args.clip_pretrained, 
            device=args.device
        )
        model.eval()
        tok = open_clip.get_tokenizer(args.clip_model)
    
    # 验证模型维度
    with torch.no_grad():
        dummy_text = tok(["test"]).to(args.device)
        dummy_embed = model.encode_text(dummy_text)
        text_dim = dummy_embed.shape[-1]
    
    print(f"Model text embedding dimension: {text_dim}")
    
    # 处理维度不匹配的情况
    if V.shape[1] != text_dim:
        print(f"\n⚠️  Dimension mismatch detected!")
        print(f"   Brain vectors: {V.shape[1]} dim")
        print(f"   Text encoder:  {text_dim} dim")
        
        if V.shape[1] == 1664 and text_dim == 1280 and args.clip_model == "ViT-bigG-14":
            print(f"\n💡 Detected MindEye brain predictions (1664 dim)")
            print(f"   These are ViT-bigG penultimate layer features (before projection)")
            print(f"\n🔄 Applying visual projection to align with text space (1280 dim)...")
            
            # 获取 visual projection 层
            if hasattr(model.visual, 'proj') and model.visual.proj is not None:
                proj_matrix = model.visual.proj.cpu()
                print(f"   Visual projection matrix shape: {proj_matrix.shape}")
                
                # ViT-bigG projection: (1664, 1280)
                if proj_matrix.shape == torch.Size([1664, 1280]):
                    V_projected = V @ proj_matrix  # [N, 1664] @ [1664, 1280] = [N, 1280]
                elif proj_matrix.shape == torch.Size([1280, 1664]):
                    V_projected = V @ proj_matrix.T
                else:
                    raise ValueError(f"Unexpected projection shape: {proj_matrix.shape}")
                
                # 归一化（与 CLIP 的输出保持一致）
                V_projected = V_projected / V_projected.norm(dim=-1, keepdim=True)
                V = V_projected
                print(f"   ✓ Successfully projected to {V.shape[1]} dim")
                print(f"   ✓ Applied L2 normalization")
            else:
                print(f"   ✗ ERROR: Visual projection layer not found in model")
                print(f"   This shouldn't happen with ViT-bigG-14")
                raise ValueError("Missing visual projection layer")
                
        else:
            print(f"\nExpected dimension combinations:")
            print(f"   1664 dim (brain) + 1280 dim (text) → MindEye predictions for ViT-bigG-14")
            print(f"   1280 dim (brain) + 1280 dim (text) → Direct ViT-bigG or ViT-H features")
            print(f"   1024 dim (brain) + 1024 dim (text) → ViT-L-14 features")
            raise ValueError(f"Cannot resolve dimension mismatch: {V.shape[1]} vs {text_dim}")
    
    print(f"\n✓ Brain vectors and text encoder dimensions aligned: {V.shape[1]} dim")
    
    tok = open_clip.get_tokenizer(args.clip_model)

    print("Encoding text library...")
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(args.device=="cuda")):
        T_embs = []
        for i in tqdm(range(0, len(caps), args.batch_size), desc="Encoding Captions"):
            batch_texts = caps[i:i+args.batch_size]
            batch_tok = tok(batch_texts).to(args.device)
            batch_T = model.encode_text(batch_tok)
            batch_T = batch_T / batch_T.norm(dim=-1, keepdim=True)
            T_embs.append(batch_T.cpu())
        T = torch.cat(T_embs, 0)
    
    print(f"Text embeddings shape: {T.shape}")
    
    print("Calculating similarities and retrieving top-K...")
    Vn = V / V.norm(dim=-1, keepdim=True)
    S = Vn @ T.T

    output_records = []
    for i in tqdm(range(V.shape[0]), desc="Retrieving"):
        if i < len(ids):
            top_indices = S[i].topk(k=min(args.topk, S.shape[1])).indices.tolist()
            output_records.append({
                "id": int(ids[i]),
                "topk": [caps[j] for j in top_indices]
            })

    print(f"Saving {len(output_records)} records to {args.out_jsonl}...")
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()