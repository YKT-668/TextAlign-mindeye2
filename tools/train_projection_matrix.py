#!/usr/bin/env python
# 训练一个线性“翻译器”，将你的1664维“脑语言”特征，精准地转换成IP-Adapter能听懂的1024维“图语言”特征。
import os
import torch
import glob
from PIL import Image
import open_clip
from tqdm import tqdm
import argparse

def main():
    ap = argparse.ArgumentParser(description="Train a linear projection matrix between two CLIP embedding spaces.")
    ap.add_argument("--image_dir", required=True, help="Directory containing images to use for training.")
    ap.add_argument("--out_pt", required=True, help="Path to save the output projection matrix .pt file.")
    
    ap.add_argument("--source_model", default="ViT-bigG-14", help="Source CLIP model name.")
    ap.add_argument("--source_pretrained", default="laion2b_s39b_b160k", help="Source model pretrained weights.")
    ap.add_argument("--source_is_penultimate", action="store_true", help="Flag if source features are from the penultimate layer (e.g., MindEye bigG).")

    ap.add_argument("--target_model", default="ViT-H-14", help="Target CLIP model name.")
    ap.add_argument("--target_pretrained", default="laion2b_s32b_b79k", help="Target model pretrained weights.")

    ap.add_argument("--max_images", type=int, default=500, help="Maximum number of images to use for training.")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    ap.add_argument("--regularization", type=float, default=1e-3, help="Lambda for Ridge Regression.")
    ap.add_argument("--device", default="cuda", help="Device to run on.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")) + glob.glob(os.path.join(args.image_dir, "*.jpg")))[:args.max_images]

    if not paths:
        raise FileNotFoundError(f"No images found in {args.image_dir} to train the projection matrix.")

    print(f"Using device: {args.device}")

    print("Loading CLIP models...")
    source_model, _, source_preprocess = open_clip.create_model_and_transforms(args.source_model, pretrained=args.source_pretrained, device=args.device)
    source_model.eval()
    
    target_model, _, target_preprocess = open_clip.create_model_and_transforms(args.target_model, pretrained=args.target_pretrained, device=args.device)
    target_model.eval()

    def encode_features(model, preprocess, ims_paths, model_name, is_penultimate, batch_size):
        features = []
        with torch.no_grad(), torch.amp.autocast(args.device, enabled=(args.device=="cuda")):
            for i in tqdm(range(0, len(ims_paths), batch_size), desc=f"Encoding with {model_name}"):
                batch_paths = ims_paths[i:i+batch_size]
                images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
                batch_im = torch.stack(images).to(args.device)
                
                if is_penultimate:
                    # --- 核心修正：手动调用视觉模型的各个部分，以获取投影前特征 ---
                    # 这是 open_clip 中 VisionTransformer 的标准前向传播路径
                    x = model.visual.conv1(batch_im)  # patch embedding
                    x = x.reshape(x.shape[0], x.shape[1], -1)
                    x = x.permute(0, 2, 1)
                    x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                    x = x + model.visual.positional_embedding.to(x.dtype)
                    x = model.visual.ln_pre(x)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = model.visual.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    
                    # 从 transformer 的输出中，我们只关心 [CLS] token 的特征
                    # 在 open_clip 中，这通常是通过一个 LayerNorm (ln_post) 来完成的
                    # 有些模型在 transformer 后直接取 [CLS] token，有些则有 pooling 操作
                    # 我们直接取第一个 token (class token)
                    feat = model.visual.ln_post(x[:, 0, :])
                else:
                    feat = model.encode_image(batch_im)

                features.append(feat.cpu().float())
        
        features = torch.cat(features, 0)
        return features

    print(f"Encoding {len(paths)} images with both models to create a training dataset...")
    X = encode_features(source_model, source_preprocess, paths, args.source_model, args.source_is_penultimate, args.batch_size)
    Y = encode_features(target_model, target_preprocess, paths, args.target_model, False, args.batch_size)

    print(f"Dataset created: X shape {X.shape}, Y shape {Y.shape}")

    print("Solving for projection matrix W using Ridge Regression...")
    try:
        W = torch.linalg.solve(X.T @ X + args.regularization * torch.eye(X.shape[1]), X.T @ Y)
    except torch.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}. The matrix might be singular.")
        print("Attempting with pseudoinverse...")
        regularized_xtx = X.T @ X + args.regularization * torch.eye(X.shape[1])
        W = torch.linalg.pinv(regularized_xtx) @ (X.T @ Y)
        print("Successfully solved using pseudoinverse.")

    print("Saving the projection matrix...")
    torch.save(W, args.out_pt)
    print(f"✓ saved W: {args.out_pt}, shape: {tuple(W.shape)}")

if __name__ == "__main__":
    main()
