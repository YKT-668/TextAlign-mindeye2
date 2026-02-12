
#!/usr/bin/env python3
# tools/train_brain2vith.py
import argparse, os, torch, json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def closed_form(X, Y, l2=0.0):
    # X: [N, D_in], Y: [N, D_out]
    # Solve W = (X^T X + λI)^-1 X^T Y
    XtX = X.T @ X
    if l2 > 0:
        XtX = XtX + l2 * torch.eye(XtX.shape[0], dtype=X.dtype, device=X.device)
    W = torch.linalg.solve(XtX, X.T @ Y)
    return W

def cosine_loss(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a * b).sum(dim=-1).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain_pt", type=str, required=True, help="N x 1664 (float32)")
    ap.add_argument("--img_vith_pt", type=str, required=True, help="N x 1024 (float32)")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path (state_dict)")
    ap.add_argument("--mode", type=str, choices=["closed_form","train"], default="closed_form")
    ap.add_argument("--lambda_l2", type=float, default=1e-3, help="ridge for closed_form / weight decay for train")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--standardize", action="store_true", help="Z-score X before fitting and store stats")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    X = torch.load(args.brain_pt, map_location="cpu").float()  # [N,1664]
    Y = torch.load(args.img_vith_pt, map_location="cpu").float()  # [N,1024]
    assert X.shape[0] == Y.shape[0], "X,Y size mismatch"
    print(f"[load] X {tuple(X.shape)}  Y {tuple(Y.shape)}")

    stats = {}
    if args.standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        X = (X - mean) / std
        stats = {"x_mean": mean, "x_std": std}
        print("[info] Standardized X")

    if args.mode == "closed_form":
        W = closed_form(X.to(device), Y.to(device), l2=args.lambda_l2)
        state = {"W": W.float().cpu()}
        state.update(stats)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(state, args.out)
        print(f"[saved] closed_form -> {args.out}  W={tuple(W.shape)}")
        return

    # train mode
    W = torch.nn.Linear(X.shape[1], Y.shape[1], bias=False).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    X, Y = None, None  # free host memory

    for ep in range(1, args.epochs + 1):
        W.train()
        total = 0.0
        for xb, yb in tqdm(dl, desc=f"train ep{ep}/{args.epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = W(xb)
            loss = cosine_loss(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"[ep{ep}] cosloss={total/len(dl.dataset):.6f}")

    state = {"W": W.weight.detach().float().cpu().t()}  # store as [1664,1024]
    state.update(stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(state, args.out)
    print(f"[saved] train -> {args.out}  W={tuple(state['W'].shape)}")

if __name__ == "__main__":
    main()
