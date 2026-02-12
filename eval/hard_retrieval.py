#二次实验时这个代码会测试当把正确图片混在 50 张长得极其相似的“干扰图”里时，你的模型还能不能根据脑信号一眼认出那张正确的原图。
import argparse, os
import numpy as np

def l2norm(x: np.ndarray, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def topk_acc(sim, k=1):
    N = sim.shape[0]
    topk = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
    correct = (topk == np.arange(N)[:, None]).any(axis=1)
    return correct.mean()

def mrr(sim):
    N = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = (order == np.arange(N)[:, None]).argmax(axis=1) + 1
    return (1.0 / ranks).mean()

def build_hard_pool(gt, M):
    sim_gt = gt @ gt.T
    np.fill_diagonal(sim_gt, -np.inf)
    # 选取每个样本最相似的 M 个作为负样本
    hard = np.argpartition(-sim_gt, kth=M-1, axis=1)[:, :M]
    return hard

def hard_retrieval(brain, gt, hard_pool):
    N = gt.shape[0]
    sim = brain @ gt.T
    
    # 创建一个全负无穷的掩码
    mask = np.full_like(sim, -np.inf, dtype=np.float32)
    rows = np.arange(N)[:, None]
    
    # 填回 hard negatives 的分数
    mask[rows, hard_pool] = sim[rows, hard_pool]
    # 填回正确答案的分数 (对角线)
    mask[np.arange(N), np.arange(N)] = sim[np.arange(N), np.arange(N)]
    return mask

def bootstrap_ci(metric_fn, sim, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = sim.shape[0]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        # 注意：这里需要传入重采样后的子矩阵
        # 严格来说 Retrieval Bootstrap 比较复杂，这里简化为对“对角线正确性”的采样
        # 为了速度和兼容性，暂且对整个 sim 矩阵行重采样
        sub_sim = sim[idx] 
        # 但列索引也变了，这里逻辑较复杂。
        # 简化版：仅对 metric 结果做 CI (二项分布近似)
        # 真正严谨的做法是对 acc 数组做 boostrap。
        # 这里为了不报错先返回 (0,0)，后续可以优化
    return 0.0, 0.0 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain_npy", required=True)
    ap.add_argument("--gt_npy", default="/mnt/work/data_cache/clip_img_gt.npy")
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # 1. 加载
    print(f"Loading Brain: {args.brain_npy}")
    brain = np.load(args.brain_npy).astype(np.float32)
    print(f"Loading GT: {args.gt_npy}")
    gt = np.load(args.gt_npy).astype(np.float32)
    
    if brain.shape != gt.shape:
        print(f"⚠️ 形状不匹配! Brain:{brain.shape} vs GT:{gt.shape}")
        # 尝试自动修正 (如果只是多了个维度)
        if brain.shape[0] == gt.shape[0]:
             brain = brain.reshape(gt.shape[0], -1)
             print(f"   -> 自动 reshape 为 {brain.shape}")

    # 2. 归一化
    brain = l2norm(brain)
    gt = l2norm(gt)

    # 3. 构建 Hard Pool
    print(f"Building Hard Pool (M={args.M})...")
    hard_pool = build_hard_pool(gt, args.M)
    
    # 4. 计算检索
    print("Computing Retrieval...")
    sim_hard = hard_retrieval(brain, gt, hard_pool)

    fwd1 = topk_acc(sim_hard, k=1)
    fwd5 = topk_acc(sim_hard, k=5)
    
    # 5. 保存
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("brain_npy,M,HardFWD@1,HardFWD@5\n")
        f.write(f"{os.path.basename(args.brain_npy)},{args.M},{fwd1:.4f},{fwd5:.4f}\n")

    print("-" * 40)
    print(f"🎯 评测结果 (M={args.M})")
    print(f"Top-1 Accuracy: {fwd1:.2%}")
    print(f"Top-5 Accuracy: {fwd5:.2%}")
    print("-" * 40)
    print(f"💾 结果已保存至: {args.out_csv}")

if __name__ == "__main__":
    main()