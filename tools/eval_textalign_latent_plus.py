#!/usr/bin/env python 
# coding: utf-8
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------------
# 基本路径
# ----------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

# 结果根目录：统一放到 cache/model_eval_results
RESULT_ROOT = os.path.join(_PROJ_ROOT, "cache", "model_eval_results")

# generative-models 里的 FrozenOpenCLIPImageEmbedder
_GEN_MODELS_DIR = os.path.join(_PROJ_ROOT, "generative-models")
if _GEN_MODELS_DIR not in sys.path:
    sys.path.append(_GEN_MODELS_DIR)

# FrozenOpenCLIPImageEmbedder 依赖较重（pytorch_lightning 等），环境缺失时回退到 open_clip。
_HAS_SGM = False
try:
    from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    _HAS_SGM = True
except Exception as e:
    FrozenOpenCLIPImageEmbedder = None
    print(f"[WARN] sgm/FrozenOpenCLIPImageEmbedder 不可用，将回退到 open_clip：{type(e).__name__}: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 你当前实验的名字（改这里就行）
exp_name   = os.environ.get("EXP_NAME", "s1_textalign_coco_train_long_v1")

# 指定本地 bigG 权重（避免联网下载）。如不存在则回退到默认 tag。
LOCAL_BIGG_BIN = os.path.join(_PROJ_ROOT, "cache", "hf_bigG", "open_clip_pytorch_model.bin")

# GT pooling 与随机评估设置
GT_POOLING = os.environ.get("GT_POOLING", "mean").strip().lower()
RAND_K = int(os.environ.get("RAND_K", "300"))
RAND_TRIALS = int(os.environ.get("RAND_TRIALS", "30"))
RAND_SEED = int(os.environ.get("RAND_SEED", "0"))

# 评测表征模式：
# - pooled: 使用 [N,1664]（现有逻辑）
# - tokens_flatten: 使用 tokens [N,256,1664] flatten -> [N, 256*1664]（论文 retrieval 口径常用）
EVAL_REPR = os.environ.get("EVAL_REPR", "pooled").strip().lower()

# 评测子集协议：
# - (default/empty): 使用全部样本（通常是 shared1000 = new_test 的 1000 unique）
# - shared982 / test982: 使用 WDS test split 的 982 unique 图像（论文 protocol 常用）
EVAL_SUBSET = os.environ.get("EVAL_SUBSET", "").strip().lower()


def _build_or_load_shared982_ids() -> np.ndarray:
    """返回 shared982 的 image ids（int64 一维数组，len=982）。

    优先读取 `${PROJ_ROOT}/src/shared982.npy`（bool mask, len=73000，True 表示该图像在 shared982 中）。
    如果不存在，则从 `src/wds/subj01/test/*.tar` 的 behav.npy 中统计 unique image indices 生成并写回该 npy。
    """
    mask_path = os.path.join(_PROJ_ROOT, "src", "shared982.npy")
    if os.path.isfile(mask_path):
        m = np.load(mask_path)
        if m.dtype == np.bool_ and m.ndim == 1:
            ids = np.where(m > 0)[0].astype(np.int64)
            return ids
        # 兼容：直接存 ids list 的情况
        if m.ndim == 1:
            return m.astype(np.int64)
        raise RuntimeError(f"shared982.npy 形状异常: {m.shape} dtype={m.dtype}")

    # Auto-generate from WDS test split (subj01).
    import glob
    try:
        import webdataset as wds
    except Exception as e:
        raise RuntimeError(
            f"EVAL_SUBSET=shared982 需要 webdataset 来自动生成 shared982.npy，但导入失败: {type(e).__name__}: {e}"
        )

    wds_root = os.path.join(_PROJ_ROOT, "src", "wds", "subj01", "test")
    urls = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not urls:
        raise RuntimeError(
            f"找不到 WDS test split shards: {wds_root}. 无法自动生成 shared982.npy"
        )

    ds = wds.WebDataset(urls).decode("torch").to_tuple("behav.npy")
    uniq = set()
    for (b,) in ds:
        # b: [1,6] 或 [?, ?, ?]，我们只取 image index
        uniq.add(int(b[0, 0]))
    ids = np.array(sorted(uniq), dtype=np.int64)
    if ids.size != 982:
        raise RuntimeError(f"从 WDS test split 得到 unique ids={ids.size}，期望 982")

    # Build bool mask compatible with shared1000.npy
    mask_len = 73000
    try:
        shared1000_path = os.path.join(_PROJ_ROOT, "src", "shared1000.npy")
        if os.path.isfile(shared1000_path):
            m1000 = np.load(shared1000_path)
            if m1000.ndim == 1 and m1000.dtype == np.bool_:
                mask_len = int(m1000.shape[0])
    except Exception:
        pass

    m = np.zeros((mask_len,), dtype=np.bool_)
    m[ids] = True
    try:
        np.save(mask_path, m)
        print(f"[SUBSET] 已自动生成 shared982 mask: {mask_path} (len={mask_len}, nnz={int(m.sum())})")
    except Exception as e:
        print(f"[WARN] 写入 shared982.npy 失败（{type(e).__name__}: {e}）；将仅在内存中使用 982 ids")
    return ids

def pool_tokens(z, pooling):
    if pooling == "cls":
        return z[:, 0, :]
    if pooling == "mean":
        return z.mean(dim=1)
    if pooling in ("patch_mean", "patch-mean"):
        return z[:, 1:, :].mean(dim=1)
    raise ValueError(f"未知 pooling: {pooling}")


def flatten_repr(x: torch.Tensor) -> torch.Tensor:
    """把特征统一变成 2D: [N,D]

    - x=[N,D] -> 原样返回
    - x=[N,T,D] -> flatten 为 [N, T*D]
    """
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.reshape(x.size(0), -1)
    raise RuntimeError(f"Unsupported feature rank: {x.dim()} shape={tuple(x.shape)}")


def encode_images_bigG_openclip(imgs: torch.Tensor, pooling: str, batch_size: int = 16, return_tokens: bool = False) -> torch.Tensor:
    """用 open_clip 的 ViT-bigG-14 编码 all_images。

    - return_tokens=False: [N,3,224,224] -> [N,1664]（按 pooling 聚合 tokens）
    - return_tokens=True : [N,3,224,224] -> [N,256,1664]（不做 pooling）

    说明：
    - 通过设置 model.visual.output_tokens=True 拿到 tokens。
    - 通过移除 model.visual.proj 获取 width=1664 的 tokens（而非默认对比学习投影维度）。
    """
    import open_clip
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )
    model = model.to(device)
    model.eval()

    # 关闭投影层：输出 transformer width=1664
    if hasattr(model, "visual") and hasattr(model.visual, "proj"):
        model.visual.proj = None

    # 开启 tokens 输出
    if hasattr(model, "visual") and hasattr(model.visual, "output_tokens"):
        model.visual.output_tokens = True

    mean = torch.tensor(OPENAI_DATASET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_DATASET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    outs = []
    imgs = imgs.float()
    with torch.no_grad():
        for i in range(0, imgs.size(0), batch_size):
            x = imgs[i:i + batch_size].to(device)
            x = (x - mean) / std
            out = model.visual(x)

            if isinstance(out, tuple) and len(out) == 2:
                pooled, tokens = out
                z = tokens if torch.is_tensor(tokens) else pooled
            else:
                z = out

            if z.dim() == 3:
                if not return_tokens:
                    z = pool_tokens(z, pooling)
            elif z.dim() == 2:
                if return_tokens:
                    raise RuntimeError(f"open_clip 未返回 tokens（得到 2D: {z.shape}），无法 tokens_flatten")
            else:
                raise RuntimeError(f"open_clip 输出维度异常: {z.shape}")

            outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0)

# ----------------------------------
# 路径设置
# ----------------------------------
exp_dir    = os.path.join(_PROJ_ROOT, "train_logs", exp_name)
infer_dir  = os.path.join(exp_dir, "inference")

brain_path = os.environ.get("BRAIN_PATH")
if not brain_path:
    brain_path = os.path.join(infer_dir, "brain_clip.pt") if os.path.isfile(os.path.join(infer_dir, "brain_clip.pt")) else (
        os.path.join(exp_dir, "brain_clip.pt") if os.path.isfile(os.path.join(exp_dir, "brain_clip.pt")) else None
    )

ids_path = os.environ.get("IDS_PATH")
if not ids_path:
    ids_path = os.path.join(infer_dir, "ids.json") if os.path.isfile(os.path.join(infer_dir, "ids.json")) else (
        os.path.join(exp_dir, "ids.json") if os.path.isfile(os.path.join(exp_dir, "ids.json")) else None
    )

# 之前评估 pipeline 时生成的 all_images.pt
_default_gt = os.path.join(_PROJ_ROOT, "evals", "all_images.pt")
if not os.path.isfile(_default_gt):
    _default_gt = os.path.join(_PROJ_ROOT, "src", "evals", "all_images.pt")
gt_path = os.environ.get("GT_PATH", _default_gt)

print("[PATH] brain:", brain_path)
print("[PATH] ids  :", ids_path)
print("[PATH] gt   :", gt_path)
print(
    f"[CFG]  eval_repr={EVAL_REPR}  gt_pooling={GT_POOLING}  rand_k={RAND_K}  rand_trials={RAND_TRIALS}  rand_seed={RAND_SEED}  eval_subset={EVAL_SUBSET or 'all'}"
)

assert brain_path is not None and os.path.isfile(brain_path), f"brain_clip.pt 不存在于 {exp_dir} 或其 inference 子目录"
assert os.path.isfile(gt_path),    f"GT 特征文件不存在: {gt_path}"

# 缓存 CLIP GT 特征的目录
os.makedirs(os.path.join(_PROJ_ROOT, "evals"), exist_ok=True)

# 本次评估结果输出目录
result_dir = os.environ.get("RESULT_DIR", os.path.join(RESULT_ROOT, exp_name))
os.makedirs(result_dir, exist_ok=True)
print("[OUT] result_dir:", result_dir)

# ======================================================================
# 1) 读取 brain_clip（pooled: [N,D]；tokens: [N,256,D]）
# ======================================================================
obj = torch.load(brain_path, map_location="cpu")
if isinstance(obj, torch.Tensor):
    brain_feats = obj
elif isinstance(obj, dict):
    cand = None
    for k, v in obj.items():
        if torch.is_tensor(v) and v.dim() in (2, 3):
            if EVAL_REPR == "pooled" and v.dim() != 2:
                continue
            if EVAL_REPR == "tokens_flatten" and v.dim() != 3:
                continue
            cand = v
            print(f"[LOAD] brain_clip: 使用 dict['{k}'] 作为特征")
            break
    assert cand is not None, (
        f"brain_clip.pt 中没有找到合适的 tensor（EVAL_REPR={EVAL_REPR}）: keys={list(obj.keys())}"
    )
    brain_feats = cand
else:
    raise RuntimeError(f"brain_clip 格式不支持: {type(obj)}")

print("[INFO] brain_feats:", brain_feats.shape)

# ======================================================================
# 2) 读取 ids.json（1000 个 global image id）
# ======================================================================
if ids_path and os.path.isfile(ids_path):
    with open(ids_path, "r") as f:
        brain_ids = json.load(f)
    brain_ids = np.asarray(brain_ids, dtype=np.int64)
    print("[INFO] brain_ids:", brain_ids.shape, "min=", brain_ids.min(), "max=", brain_ids.max())
else:
    brain_ids = np.arange(brain_feats.shape[0], dtype=np.int64)
    print("[WARN] 未提供 ids.json，默认使用 [0..N-1] 作为 brain_ids")

# ======================================================================
# 3) 读取 GT（all_images.pt），如有必要先用 CLIP 编成特征
# ======================================================================
gt_obj = torch.load(gt_path, map_location="cpu")

gt_feats = None
gt_ids   = None

if isinstance(gt_obj, torch.Tensor):
    # 情况 A：已经是 [N, D] 的特征
    if gt_obj.dim() == 2:
        if EVAL_REPR == "tokens_flatten":
            raise RuntimeError(f"EVAL_REPR=tokens_flatten 需要 GT tokens [N,256,1664]，但 GT 是 2D: {tuple(gt_obj.shape)}")
        gt_feats = gt_obj
        print("[GT] 直接使用 tensor 作为特征，shape =", gt_feats.shape)
        if GT_POOLING != "cls":
            print("[WARN] GT 已是 2D 特征，pooling 不会生效；请确认特征来源与目标 pooling 一致")

    # 情况 B：是 [N, 3, 224, 224] 的图像，需要过一遍 CLIP
    elif gt_obj.dim() == 4 and gt_obj.shape[1] == 3:
        # 如果是原始图像，优先尝试离线缓存特征文件，避免再次下载大权重
        pool_tag = GT_POOLING.replace("-", "_")
        offline_feat_path = os.path.join(_PROJ_ROOT, "evals", f"all_images_bigG_1664_{pool_tag}.pt")
        offline_tok_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_tokens_256x1664.pt")
        legacy_feat_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_1664.pt")
        imgs = gt_obj.float()   # [N,3,224,224]
        if EVAL_REPR == "tokens_flatten":
            if os.path.isfile(offline_tok_path):
                gt_feats = torch.load(offline_tok_path, map_location="cpu")
                assert gt_feats.dim() == 3 and gt_feats.shape[0] == imgs.shape[0], (
                    f"缓存 tokens 形状不匹配: {gt_feats.shape} vs {imgs.shape[0]}"
                )
                print(f"[GT] 发现离线 tokens 文件 {offline_tok_path}，直接加载，shape = {gt_feats.shape}")
            else:
                print("[GT] 需要 tokens_flatten：编码并缓存 GT tokens，shape =", imgs.shape)
                if _HAS_SGM:
                    raise RuntimeError("当前环境 sgm 不可用；请用 open_clip fallback 生成 GT tokens（保持 _HAS_SGM=False）")
                print("[CLIP] 使用 open_clip 生成 tokens（无 pytorch_lightning 依赖）")
                gt_feats = encode_images_bigG_openclip(imgs, GT_POOLING, batch_size=16, return_tokens=True)
                print("[GT] tokens 编码完成, shape =", gt_feats.shape, " -> 保存到离线文件")
                try:
                    os.makedirs(os.path.dirname(offline_tok_path), exist_ok=True)
                    torch.save(gt_feats, offline_tok_path)
                    print(f"[GT] 离线 tokens 已保存: {offline_tok_path}")
                except Exception as e:
                    print(f"[WARN] 保存离线 tokens 失败: {e}")
        else:
            if os.path.isfile(offline_feat_path):
                gt_feats = torch.load(offline_feat_path, map_location="cpu")
                assert gt_feats.dim() == 2 and gt_feats.shape[0] == imgs.shape[0], (
                    f"缓存特征形状不匹配: {gt_feats.shape} vs {imgs.shape[0]}"
                )
                print(f"[GT] 发现离线特征文件 {offline_feat_path}，直接加载，shape = {gt_feats.shape}")
            elif GT_POOLING == "cls" and os.path.isfile(legacy_feat_path):
                gt_feats = torch.load(legacy_feat_path, map_location="cpu")
                assert gt_feats.dim() == 2 and gt_feats.shape[0] == imgs.shape[0], (
                    f"缓存特征形状不匹配: {gt_feats.shape} vs {imgs.shape[0]}"
                )
                print(f"[GT] 发现离线特征文件 {legacy_feat_path}，直接加载，shape = {gt_feats.shape}")
            else:
                print("[GT] 检测到原始图像，shape =", imgs.shape, " -> 使用 CLIP 编码为 1664 维特征 (首次生成缓存)")
                if _HAS_SGM:
                    raise RuntimeError("当前环境 sgm 不可用；请用 open_clip fallback 生成 GT pooled 特征（保持 _HAS_SGM=False）")
                print("[CLIP] 使用 open_clip 回退路径（无 pytorch_lightning 依赖）")
                gt_feats = encode_images_bigG_openclip(imgs, GT_POOLING, batch_size=16, return_tokens=False)
                print("[GT] CLIP 特征编码完成, shape =", gt_feats.shape, " -> 保存到离线文件")
                try:
                    os.makedirs(os.path.dirname(offline_feat_path), exist_ok=True)
                    torch.save(gt_feats, offline_feat_path)
                    print(f"[GT] 离线特征已保存: {offline_feat_path}")
                except Exception as e:
                    print(f"[WARN] 保存离线特征失败: {e}")

    # 情况 C：tokens
    elif gt_obj.dim() == 3:
        if EVAL_REPR == "tokens_flatten":
            gt_feats = gt_obj
            print(f"[GT] tokens_flatten：直接使用 tokens，shape = {tuple(gt_feats.shape)}")
        else:
            gt_feats = pool_tokens(gt_obj, GT_POOLING)
            print(f"[GT] 检测到 tokens，pooling={GT_POOLING}，shape =", gt_feats.shape)

    else:
        raise RuntimeError(f"GT tensor 形状不支持: {gt_obj.shape}")

elif isinstance(gt_obj, dict):
    # dict 情况：先找特征
    for key in ["features", "feats", "clip_feats", "img_feats"]:
        if key in gt_obj and torch.is_tensor(gt_obj[key]):
            gt_feats = gt_obj[key]
            print(f"[GT] 使用 dict['{key}'] 作为特征，shape =", gt_feats.shape)
            break
    if gt_feats is None:
        for k, v in gt_obj.items():
            if torch.is_tensor(v) and v.dim() == 2 and EVAL_REPR != "tokens_flatten":
                gt_feats = v
                print(f"[GT] 兜底使用 dict['{k}'] 作为特征，shape =", gt_feats.shape)
                if GT_POOLING != "cls":
                    print("[WARN] GT 已是 2D 特征，pooling 不会生效；请确认特征来源与目标 pooling 一致")
                break
            if torch.is_tensor(v) and v.dim() == 3:
                if EVAL_REPR == "tokens_flatten":
                    gt_feats = v
                    print(f"[GT] tokens_flatten：兜底使用 dict['{k}'] 作为 tokens，shape =", gt_feats.shape)
                else:
                    gt_feats = pool_tokens(v, GT_POOLING)
                    print(f"[GT] 兜底使用 dict['{k}'] 作为 tokens，pooling={GT_POOLING}，shape =", gt_feats.shape)
                break
    assert gt_feats is not None, f"GT 文件中没有找到 2D 特征 tensor: keys={list(gt_obj.keys())}"

    # 再找 ids
    for key in ["ids", "image_ids", "img_ids", "nsd_ids"]:
        if key in gt_obj:
            gt_ids = np.asarray(gt_obj[key], dtype=np.int64)
            print(f"[GT] 使用 dict['{key}'] 作为 image_ids，shape =", gt_ids.shape)
            break
else:
    raise RuntimeError(f"GT 文件格式不支持: {type(gt_obj)}")

# 如果没有单独的 gt_ids，就假设顺序与 brain_ids 一致
if gt_ids is None:
    # 优先用 shared1000.npy 推断 canonical 的 1000 张图像 id 顺序，避免仅按位置对齐。
    shared_path = os.path.join(_PROJ_ROOT, "src", "shared1000.npy")
    if os.path.isfile(shared_path):
        try:
            shared_mask = np.load(shared_path)
            shared_ids = np.where(shared_mask > 0)[0].astype(np.int64)
            if len(shared_ids) == int(gt_feats.shape[0]):
                gt_ids = shared_ids
                print(f"[GT] 未检测到 ids 字段，使用 shared1000.npy 推断 gt_ids，len={len(gt_ids)}")
            else:
                gt_ids = brain_ids.copy()
                print(
                    f"[WARN] shared1000.npy 推断得到 {len(shared_ids)} ids，但 GT N={gt_feats.shape[0]}；将回退为按 brain_ids 顺序对齐"
                )
        except Exception as e:
            gt_ids = brain_ids.copy()
            print(f"[WARN] 读取 shared1000.npy 失败（{type(e).__name__}: {e}）；将回退为按 brain_ids 顺序对齐")
    else:
        gt_ids = brain_ids.copy()
        print("[GT] 未检测到单独的 ids 字段，假设顺序与 brain_ids 一致")

# ======================================================================
# 4) 按 ids 对齐顺序（保证 brain_feats[i] 和 gt_feats[i] 是同一张图）
# ======================================================================
id2row = {int(gid): i for i, gid in enumerate(gt_ids)}
rows = []
for gid in brain_ids:
    if int(gid) not in id2row:
        raise KeyError(f"GT 中找不到 image_id={int(gid)}，请检查 GT 特征文件的 ids 是否完整")
    rows.append(id2row[int(gid)])
rows = np.asarray(rows, dtype=np.int64)

gt_sel = gt_feats[rows]
print("[INFO] gt_sel:", gt_sel.shape)

# ======================================================================
# 4.1) 可选：按 protocol 子集过滤（shared982）
# ======================================================================
if EVAL_SUBSET in ("shared982", "test982", "wds_test_982", "wds-test-982"):
    subset_ids = _build_or_load_shared982_ids()
    keep = np.isin(brain_ids, subset_ids)
    kept_n = int(keep.sum())
    if kept_n != int(subset_ids.shape[0]):
        print(
            f"[WARN] shared982 期望保留 {int(subset_ids.shape[0])}，但在 brain_ids 中匹配到 {kept_n}。"
            " 这通常意味着当前 brain_ids 不来自 shared1000/new_test，或 ids 口径不一致。"
        )
    if kept_n <= 0:
        raise RuntimeError("EVAL_SUBSET=shared982 过滤后样本数为 0，无法评测")
    brain_feats = brain_feats[keep]
    gt_sel = gt_sel[keep]
    brain_ids = brain_ids[keep]
    print(f"[SUBSET] applied {EVAL_SUBSET}: kept {kept_n} samples")

if EVAL_REPR == "pooled":
    assert brain_feats.shape == gt_sel.shape, \
        f"特征维度不一致: brain {brain_feats.shape} vs gt {gt_sel.shape}"
elif EVAL_REPR == "tokens_flatten":
    assert brain_feats.dim() == 3 and gt_sel.dim() == 3, (
        f"tokens_flatten 需要 3D tokens: brain {brain_feats.shape} vs gt {gt_sel.shape}"
    )
else:
    raise ValueError(f"Unknown EVAL_REPR: {EVAL_REPR}")

# ======================================================================
# 5) 归一化并计算相似度矩阵
# ======================================================================
brain_feats = brain_feats.to(device=device, dtype=torch.float32)
gt_sel      = gt_sel.to(device=device, dtype=torch.float32)

if EVAL_REPR == "tokens_flatten":
    brain_feats = flatten_repr(brain_feats)
    gt_sel = flatten_repr(gt_sel)

brain_n = F.normalize(brain_feats, dim=-1)
gt_n    = F.normalize(gt_sel,     dim=-1)

# sim[i, j] = cos(z_brain[i], z_img[j])
sim = brain_n @ gt_n.t()      # [N, N]
labels = torch.arange(sim.size(0), device=sim.device)

# 对角线相似度（CLIP latent 对齐程度）
sim_diag = torch.diag(sim)    # [N]

# ======================================================================
# 5.1) 300-way 随机评估（可选）
# ======================================================================
def random_k_eval(sim_mat, labels, k=300, trials=30, seed=0):
    if k <= 0 or trials <= 0:
        return None
    n = sim_mat.size(0)
    if k > n:
        raise ValueError(f"random_k_eval: k={k} > N={n}")
    sim_cpu = sim_mat.detach().cpu()
    rng = np.random.default_rng(seed)
    top1_list = []
    top5_list = []
    for _ in range(trials):
        correct1 = 0
        correct5 = 0
        for i in range(n):
            neg = rng.choice(n - 1, size=k - 1, replace=False)
            neg = np.where(neg >= i, neg + 1, neg)
            cand = np.concatenate(([i], neg))
            cand_t = torch.from_numpy(cand).long()
            vals = sim_cpu[i, cand_t]
            tk = min(5, k)
            topk_idx = torch.topk(vals, k=tk).indices
            if topk_idx[0].item() == 0:
                correct1 += 1
            if (topk_idx == 0).any().item():
                correct5 += 1
        top1_list.append(correct1 / n)
        top5_list.append(correct5 / n)
    return {
        "top1_mean": float(np.mean(top1_list)),
        "top1_std": float(np.std(top1_list)),
        "top5_mean": float(np.mean(top5_list)),
        "top5_std": float(np.std(top5_list)),
    }

# ======================================================================
# 6) 各种指标计算
# ======================================================================
def topk_acc(sim_mat, labels, k=1):
    topk = sim_mat.topk(k, dim=-1).indices   # [N, k]
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct

def rank_stats(sim_mat, labels):
    """
    返回：
      ranks: 每个样本的 rank（1=最优）
      recall_at: dict{k: recall@k}
      mean_rank, median_rank
    """
    # sim_mat: [N, N]，越大越相似
    sorted_idx = sim_mat.argsort(dim=1, descending=True)  # [N, N]
    # sorted_idx[i] 中找 label i 所在的位置
    matches = (sorted_idx == labels.unsqueeze(1))         # [N, N] bool
    # 每行恰好有一个 True，argmax 给出索引
    ranks = matches.float().argmax(dim=1) + 1            # [N] in [1..N]

    ranks_np = ranks.detach().cpu().numpy()
    mean_rank   = float(ranks_np.mean())
    median_rank = float(np.median(ranks_np))

    recall_at = {}
    for k in [1, 5, 10, 20, 50]:
        recall_at[k] = float((ranks_np <= k).mean())

    return ranks, recall_at, mean_rank, median_rank

# FWD: brain -> image
top1_fwd = topk_acc(sim, labels, k=1)
top5_fwd = topk_acc(sim, labels, k=5)
ranks_fwd, recall_fwd, mean_rank_fwd, median_rank_fwd = rank_stats(sim, labels)

# BWD: image -> brain
sim_T = sim.t()
top1_bwd = topk_acc(sim_T, labels, k=1)
top5_bwd = topk_acc(sim_T, labels, k=5)
ranks_bwd, recall_bwd, mean_rank_bwd, median_rank_bwd = rank_stats(sim_T, labels)

rand_fwd = random_k_eval(sim, labels, k=RAND_K, trials=RAND_TRIALS, seed=RAND_SEED)
rand_bwd = random_k_eval(sim_T, labels, k=RAND_K, trials=RAND_TRIALS, seed=RAND_SEED + 1)

# CLIP(latent) 相似度统计
sim_diag_np = sim_diag.detach().cpu().numpy()
clip_latent_mean   = float(sim_diag_np.mean())
clip_latent_std    = float(sim_diag_np.std())
clip_latent_p25    = float(np.percentile(sim_diag_np, 25))
clip_latent_p50    = float(np.percentile(sim_diag_np, 50))
clip_latent_p75    = float(np.percentile(sim_diag_np, 75))

# ======================================================================
# 7) 打印 & 保存数值结果
# ======================================================================
print("====================================================")
print("TextAlign latent eval (单被试 latent 检索 + CLIP(latent))")
print(f"[Retrieval FWD] Top-1: {top1_fwd*100:.2f}%   Top-5: {top5_fwd*100:.2f}%")
print(f"[Retrieval BWD] Top-1: {top1_bwd*100:.2f}%   Top-5: {top5_bwd*100:.2f}%")
print(f"[FWD]  mean rank = {mean_rank_fwd:.2f}, median rank = {median_rank_fwd:.2f}")
print(f"[BWD]  mean rank = {mean_rank_bwd:.2f}, median rank = {median_rank_bwd:.2f}")
print("[FWD]  Recall@K:", {k: f"{v*100:.2f}%" for k, v in recall_fwd.items()})
print("[BWD]  Recall@K:", {k: f"{v*100:.2f}%" for k, v in recall_bwd.items()})
print(f"[CLIP(latent)] mean={clip_latent_mean:.4f}, std={clip_latent_std:.4f}, "
      f"p25={clip_latent_p25:.4f}, p50={clip_latent_p50:.4f}, p75={clip_latent_p75:.4f}")
print(f"[ALIGN-CHECK] sim.diag().mean={clip_latent_mean:.4f}  sim.mean={float(sim.detach().cpu().mean()):.4f}")
if rand_fwd is not None and rand_bwd is not None:
    print(f"[Random {RAND_K}-way x{RAND_TRIALS}] FWD Top-1: {rand_fwd['top1_mean']*100:.2f}% "
          f"(±{rand_fwd['top1_std']*100:.2f}%)  Top-5: {rand_fwd['top5_mean']*100:.2f}% "
          f"(±{rand_fwd['top5_std']*100:.2f}%)")
    print(f"[Random {RAND_K}-way x{RAND_TRIALS}] BWD Top-1: {rand_bwd['top1_mean']*100:.2f}% "
          f"(±{rand_bwd['top1_std']*100:.2f}%)  Top-5: {rand_bwd['top5_mean']*100:.2f}% "
          f"(±{rand_bwd['top5_std']*100:.2f}%)")
print("====================================================")

# 保存 metrics.json
metrics = {
    "exp_name": exp_name,
    "N": int(sim.size(0)),
    "eval_repr": EVAL_REPR,
    "gt_pooling": GT_POOLING,
    "rand_k": RAND_K,
    "rand_trials": RAND_TRIALS,
    "rand_seed": RAND_SEED,
    "retrieval": {
        "fwd": {
            "top1": top1_fwd,
            "top5": top5_fwd,
            "mean_rank": mean_rank_fwd,
            "median_rank": median_rank_fwd,
            "recall_at": recall_fwd,
        },
        "bwd": {
            "top1": top1_bwd,
            "top5": top5_bwd,
            "mean_rank": mean_rank_bwd,
            "median_rank": median_rank_bwd,
            "recall_at": recall_bwd,
        },
    },
    "clip_latent": {
        "mean": clip_latent_mean,
        "std": clip_latent_std,
        "p25": clip_latent_p25,
        "p50": clip_latent_p50,
        "p75": clip_latent_p75,
    },
}
if rand_fwd is not None and rand_bwd is not None:
    metrics["random_retrieval"] = {
        "k": RAND_K,
        "trials": RAND_TRIALS,
        "fwd": rand_fwd,
        "bwd": rand_bwd,
    }

with open(os.path.join(result_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
    f.write("=== TextAlign latent eval ===\n")
    f.write(json.dumps(metrics, indent=2))
    f.write("\n")

# ======================================================================
# 8) 可视化：直方图 & 检索曲线 & 热力图
# ======================================================================

# 8.1 CLIP(latent) 对角线相似度直方图
plt.figure(figsize=(5,4))
plt.hist(sim_diag_np, bins=40, alpha=0.8)
plt.xlabel("cosine(sim_brain, sim_image)  (diagonal)")
plt.ylabel("count")
plt.title(f"CLIP(latent) diag cosine\nexp={exp_name}")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "sim_diag_hist.png"), dpi=200)
plt.close()

# 8.2 FWD/BWD rank 直方图（log 纵轴方便看长尾）
def save_rank_hist(ranks, title, fname):
    ranks_np = ranks.detach().cpu().numpy()
    plt.figure(figsize=(5,4))
    plt.hist(ranks_np, bins=50, log=True)
    plt.xlabel("rank (1 = best)")
    plt.ylabel("count (log)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, fname), dpi=200)
    plt.close()

save_rank_hist(ranks_fwd, f"FWD rank distribution\nexp={exp_name}", "rank_hist_fwd.png")
save_rank_hist(ranks_bwd, f"BWD rank distribution\nexp={exp_name}", "rank_hist_bwd.png")

# 8.3 Retrieval Recall@K 曲线（FWD/BWD 各一条）
Ks = [1, 5, 10, 20, 50]
fwd_vals = [recall_fwd[k]*100 for k in Ks]
bwd_vals = [recall_bwd[k]*100 for k in Ks]

plt.figure(figsize=(5,4))
plt.plot(Ks, fwd_vals, marker="o", label="FWD")
plt.plot(Ks, bwd_vals, marker="s", label="BWD")
plt.xlabel("K")
plt.ylabel("Recall@K (%)")
plt.title(f"Retrieval Recall@K\nexp={exp_name}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "retrieval_curve.png"), dpi=200)
plt.close()

# 8.4 相似度矩阵热力图（抽一个 64×64 子块）
N = sim.size(0)
if N >= 64:
    # 为了视觉效果，随机取 64 个样本（也可以取前 64 个）
    idx = torch.randperm(N)[:64]
    sim_sub = sim[idx][:, idx].detach().cpu().numpy()
    plt.figure(figsize=(5,4))
    plt.imshow(sim_sub, vmin=-1, vmax=1, cmap="viridis")
    plt.colorbar(label="cosine similarity")
    plt.title(f"Similarity heatmap (64x64 subset)\nexp={exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "sim_heatmap_64x64.png"), dpi=200)
    plt.close()

print(f"[DONE] 所有指标与图像已保存到: {result_dir}")
