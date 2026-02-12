import torch
import torch.nn.functional as F

B_PATH = "runs/subj01_inference_run_final/decoded_features/brain_clip_vectors.pt"
T_PATH = "runs/subj01_inference_run_final/eval_results/llm_teacher_clip.pt"
W_PATH = "checkpoints/brain2teacher_subj01.pt"

print("[load] brain =", B_PATH)
print("[load] teacher =", T_PATH)
print("[load] W =", W_PATH)

B = torch.load(B_PATH, map_location="cpu")     # (N, 1664)
T = torch.load(T_PATH, map_location="cpu")     # (N, 768)

ckpt = torch.load(W_PATH, map_location="cpu")
W = ckpt["W"].float() if isinstance(ckpt, dict) and "W" in ckpt else ckpt.float()

B_proj = B @ W                                 # (N, 768)

Bn = F.normalize(B_proj, dim=-1)
Tn = F.normalize(T, dim=-1)
S = Bn @ Tn.T                                  # (N, N)

N = S.size(0)
diag_cos = S.diag().mean().item()
top1 = (S.argmax(dim=1) == torch.arange(N)).float().mean().item()
top5 = (
    (torch.topk(S, k=5, dim=1).indices == torch.arange(N).unsqueeze(1))
    .any(dim=1)
    .float()
    .mean()
    .item()
)

print(f"[brain→LLM teacher] diag cosine mean = {diag_cos:.4f}")
print(f"[brain→LLM teacher] top1 = {top1*100:.2f}%")
print(f"[brain→LLM teacher] top5 = {top5*100:.2f}%")
