#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import random
import re
import threading
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import httpx
import urllib3
from tqdm import tqdm

# 禁用自签名证书警告（针对 IP 直连模式）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. Filters / Heuristics (keep prompt unchanged; filters are local)
# ==========================================

FORBIDDEN = {"not", "no", "never", "instead", "wrong", "but", "however", "rather than", "without"}
NEGATION_RE = re.compile(r"\b(?:not|no|never|without)\b", flags=re.IGNORECASE)
UNCERTAIN_RE = re.compile(
    r"\b(?:this\s+image|looks\s+like|possibly|maybe|might|appears\s+to|seems\s+to|could\s+be)\b",
    flags=re.IGNORECASE,
)

def has_forbidden_words(s: str) -> bool:
    low = " " + (s or "").lower().strip() + " "
    for w in FORBIDDEN:
        if f" {w} " in low:
            return True
    return False

def safe_word_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([w for w in re.split(r"\s+", s) if w])

def length_ok(pos: str, neg: str, max_ratio_delta: float = 0.30) -> bool:
    p = max(1, safe_word_count(pos))
    n = max(1, safe_word_count(neg))
    ratio = n / p
    return (1.0 - max_ratio_delta) <= ratio <= (1.0 + max_ratio_delta)

def token_overlap_ok(pos: str, neg: str, min_jaccard: float = 0.20) -> bool:
    pos_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", pos or "")}
    neg_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", neg or "")}
    if not pos_words or not neg_words:
        return False
    inter = len(pos_words & neg_words)
    union = len(pos_words | neg_words)
    return (inter / max(1, union)) >= float(min_jaccard)

# ==========================================
# 1. Network & API (thread-local client reuse)
# ==========================================

_thread_local = threading.local()

def deepseek_client():
    """
    Uses openai python client with direct IP connection to bypass DNS/Proxy issues.
    Thread-local reuse: each worker thread constructs one OpenAI+httpx client once.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Please `pip install openai`")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing env DEEPSEEK_API_KEY")

    # Direct IP + Host masquerade
    target_ip_url = "https://116.205.40.114/v1"
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()

    # thread-local cache
    if getattr(_thread_local, "client", None) is not None and getattr(_thread_local, "model", None) is not None:
        return _thread_local.client, _thread_local.model

    http_client = httpx.Client(
        verify=False,          # 忽略 SSL 证书
        trust_env=False,       # 忽略系统/Conda 代理
        headers={"Host": "api.deepseek.com"},
        timeout=120.0
    )

    client = OpenAI(
        api_key=api_key,
        base_url=target_ip_url,
        http_client=http_client
    )

    _thread_local.client = client
    _thread_local.model = model
    return client, model

def build_prompt(pos_caption: str) -> str:

    return (

        "You are helping me generate stimulus-preserving counterfactual hard-negative captions for contrastive learning in brain decoding.\n\n"

        "Given ONE ground-truth caption describing an image, generate a set of counterfactual captions that are:\n"
        "(1) stimulus-preserving: they must still describe the SAME overall scene as the ground-truth (same setting, viewpoint, action, and global context);\n"
        "(2) critically wrong: each caption must contain EXACTLY ONE targeted semantic deviation that would make it incorrect for the image;\n"
        "(3) near-miss: the deviation should be subtle and plausible (hard negative), not obviously wrong or absurd;\n"
        '(4) implicit: DO NOT use explicit negation or correction language (forbidden words include: "not", "no", "never", "instead", "wrong", "but", "however", "rather than", "without"). You must phrase it as an alternative hypothesis, like a natural caption.\n\n'
        "You must generate 12 candidates total with the following coverage:\n"
        "- 4 Object counterfactuals: change the identity/category of ONE main object to a visually plausible alternative (e.g., bird→squirrel, dog→wolf, cup→bowl), while keeping everything else consistent.\n"
        "- 4 Attribute counterfactuals: change ONE attribute of ONE object (color/material/number/size/age) while keeping object identity and scene consistent.\n"
        "- 4 Relation counterfactuals: change ONE relationship/spatial relation/action relation (left/right, in/on/under, holding/wearing/next to) while keeping objects and setting consistent.\n\n"
        "Hardness constraints (must follow all):\n"
        "- Make the counterfactual captions highly similar in wording to the ground-truth caption (paraphrase minimally; keep most tokens unchanged).\n"
        "- Keep the sentence length and style similar to the ground-truth caption.\n"
        "- Keep the same number of main entities as much as possible; do not introduce new objects unless you are replacing exactly one object.\n"
        "- Preserve the global scene semantics (place, activity, camera/viewpoint, background) as much as possible.\n"
        "- Each candidate must differ from the ground-truth by ONE and ONLY ONE key semantic edit. Avoid multiple edits.\n"
        "- Avoid trivial edits like adding/removing generic adjectives that do not change meaning.\n\n"
        "Return the result as STRICT JSON (no markdown), a list of 12 items.\n"
        "Each item must have:\n"
        '{\n'
        '  "type": "object" | "attribute" | "relation",\n'
        '  "edit": "what single semantic detail you changed (short phrase)",\n'
        '  "caption_cf": "the counterfactual caption (one sentence, natural language)",\n'
        '  "changed_span": "the exact phrase in caption_cf that contains the changed detail"\n'
        "}\n\n"
        f'Ground-truth caption:\n"{pos_caption}"\n'

    )

def _extract_json_loose(s: str) -> Any:
    """
    Robust JSON extraction:
    - Try direct json.loads
    - If fails, try to extract first {...} or [...] span
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty response")

    try:
        return json.loads(s)
    except Exception:
        pass

    # Try extract list
    lb = s.find("[")
    rb = s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            return json.loads(s[lb:rb + 1])
        except Exception:
            pass

    # Try extract object
    lo = s.find("{")
    ro = s.rfind("}")
    if lo != -1 and ro != -1 and ro > lo:
        try:
            return json.loads(s[lo:ro + 1])
        except Exception:
            pass

    # last resort
    raise ValueError("Could not parse JSON from response")

def _parse_items(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - list[dict]
      - {"items":[...]} or {"candidates":[...]} or any dict containing a list value
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("items", "candidates"):
            v = obj.get(key, None)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        for v in obj.values():
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []

def _clean_candidates(pos_caption: str,
                      items: List[Dict[str, Any]],
                      max_ratio_delta: float = 0.30,
                      min_jaccard: float = 0.20,
                      forbid_uncertain: bool = True) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        t = str(it.get("type", "")).strip().lower()
        cap = str(it.get("caption_cf", "")).strip()
        if t not in ("object", "attribute", "relation"):
            continue
        if not cap:
            continue
        if has_forbidden_words(cap) or NEGATION_RE.search(cap):
            continue
        if forbid_uncertain and UNCERTAIN_RE.search(cap):
            continue
        if not length_ok(pos_caption, cap, max_ratio_delta=max_ratio_delta):
            continue
        if not token_overlap_ok(pos_caption, cap, min_jaccard=min_jaccard):
            continue
        cleaned.append({
            "type": t,
            "edit": str(it.get("edit", "")).strip(),
            "caption_cf": cap,
            "changed_span": str(it.get("changed_span", "")).strip(),
        })
    return cleaned

def call_deepseek_json(pos_caption: str,
                       max_retries: int = 6,
                       sleep_base: float = 1.5,
                       min_keep: int = 6,
                       max_ratio_delta: float = 0.30,
                       min_jaccard: float = 0.20,
                       forbid_uncertain: bool = True) -> List[Dict[str, Any]]:
    client, model = deepseek_client()

    # keep prompt unchanged; system message is allowed
    system_prompt = "You are a careful assistant. Output JSON only."
    user_prompt = build_prompt(pos_caption)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Keep json_object for stability; parser accepts list or dict anyway
                response_format={"type": "json_object"},
                max_tokens=1800,
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            obj = _extract_json_loose(content)
            items = _parse_items(obj)
            cleaned = _clean_candidates(
                pos_caption=pos_caption,
                items=items,
                max_ratio_delta=max_ratio_delta,
                min_jaccard=min_jaccard,
                forbid_uncertain=forbid_uncertain,
            )
            if len(cleaned) < min_keep:
                raise ValueError(f"Too few cleaned candidates: {len(cleaned)}")
            return cleaned
        except Exception as e:
            last_err = e
            wait = min(sleep_base * (2 ** attempt) + 0.1 * attempt, 30)
            time.sleep(wait)

    # all retries failed
    return []

# ==========================================
# 2. Encoder backends + auto-select to match teacher space
# ==========================================

class TextEncoderBackend:
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device

    def dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, texts: List[str], batch: int = 64) -> torch.Tensor:
        raise NotImplementedError

class OpenCLIPBackend(TextEncoderBackend):
    def __init__(self, model_name: str, pretrained: str, device: str):
        super().__init__(name=f"open_clip:{model_name}:{pretrained}", device=device)
        import open_clip  # type: ignore
        self._open_clip = open_clip
        self._model_name = model_name
        self._pretrained = pretrained
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model = model.float().eval()

        # infer dim by encoding empty
        with torch.no_grad():
            tok = self._tokenizer(["a photo"]).to(device)
            v = self._model.encode_text(tok)
            self._dim = int(v.shape[-1])

    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch):
            tt = texts[i:i + batch]
            tok = self._tokenizer(tt).to(self.device)
            v = self._model.encode_text(tok)
            v = F.normalize(v.float(), dim=-1)
            feats.append(v.detach().cpu())
        return torch.cat(feats, dim=0)

class TransformersCLIPGetTextFeaturesBackend(TextEncoderBackend):
    def __init__(self, model_name: str, device: str):
        super().__init__(name=f"hf_clip_get_text_features:{model_name}", device=device)
        from transformers import CLIPModel, CLIPTokenizer  # type: ignore
        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(device).eval().float()

        with torch.no_grad():
            enc = self._tokenizer(["a photo"], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            v = self._model.get_text_features(**enc)
            self._dim = int(v.shape[-1])

    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch):
            tt = texts[i:i + batch]
            enc = self._tokenizer(tt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            v = self._model.get_text_features(**enc)
            v = F.normalize(v.float(), dim=-1)
            feats.append(v.detach().cpu())
        return torch.cat(feats, dim=0)

class TransformersCLIPPoolerBackend(TextEncoderBackend):
    def __init__(self, model_name: str, device: str):
        super().__init__(name=f"hf_clip_pooler_output:{model_name}", device=device)
        from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore
        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._text_model = CLIPTextModel.from_pretrained(model_name).to(device).eval().float()

        with torch.no_grad():
            enc = self._tokenizer(["a photo"], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            out = self._text_model(**enc)
            v = out.pooler_output
            self._dim = int(v.shape[-1])

    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch):
            tt = texts[i:i + batch]
            enc = self._tokenizer(tt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            out = self._text_model(**enc)
            v = out.pooler_output
            v = F.normalize(v.float(), dim=-1)
            feats.append(v.detach().cpu())
        return torch.cat(feats, dim=0)

class OpenAIClipBackend(TextEncoderBackend):
    def __init__(self, device: str):
        super().__init__(name="openai_clip:ViT-L/14", device=device)
        import clip  # type: ignore
        self._clip = clip
        model, _ = clip.load("ViT-L/14", device=device)
        self._model = model.float().eval()

        with torch.no_grad():
            tok = clip.tokenize(["a photo"], truncate=True).to(device)
            v = self._model.encode_text(tok)
            self._dim = int(v.shape[-1])

    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch):
            tt = texts[i:i + batch]
            tok = self._clip.tokenize(tt, truncate=True).to(self.device)
            v = self._model.encode_text(tok)
            v = F.normalize(v.float(), dim=-1)
            feats.append(v.detach().cpu())
        return torch.cat(feats, dim=0)

def build_candidate_backends(device: str) -> List[TextEncoderBackend]:
    backends: List[TextEncoderBackend] = []

    # 1. Try openai/clip python package (本地缓存，最稳)
    try:
        backends.append(OpenAIClipBackend(device=device))
        # 【关键修改】：只要这个成功了，立刻返回！别去试后面那些联网的了！
        print(f"[Encoder] OpenAI Clip loaded successfully. Skipping others to avoid network hang.")
        return backends 
    except Exception as e:
        print(f"[Encoder] Failed to load OpenAI Clip: {e}")
        pass

    # --- 如果上面那个失败了，才会走到下面这些（作为备用） ---

    # Try OpenCLIP bigG
    try:
        pretrained = os.environ.get("OPENCLIP_PRETRAINED", "laion2b_s39b_b160k").strip()
        backends.append(OpenCLIPBackend(model_name="ViT-bigG-14", pretrained=pretrained, device=device))
    except Exception:
        pass

    # Try OpenCLIP ViT-L-14 openai
    try:
        backends.append(OpenCLIPBackend(model_name="ViT-L-14", pretrained="openai", device=device))
    except Exception:
        pass

    # Try HF CLIP (Transformers)
    try:
        backends.append(TransformersCLIPGetTextFeaturesBackend(model_name="openai/clip-vit-large-patch14", device=device))
    except Exception:
        pass
    
    try:
        backends.append(TransformersCLIPPoolerBackend(model_name="openai/clip-vit-large-patch14", device=device))
    except Exception:
        pass

    return backends

def auto_select_backend(teacher_feats: torch.Tensor,
                        sample_captions: List[str],
                        device: str) -> TextEncoderBackend:
    """
    Choose the backend that best matches teacher_feats space.
    Match criterion: mean cosine similarity between encoded(pos_caption) and teacher_feats.
    """
    teacher_feats = teacher_feats.float()
    teacher_feats = F.normalize(teacher_feats, dim=-1)

    D = int(teacher_feats.shape[-1])
    backends = build_candidate_backends(device=device)
    if len(backends) == 0:
        raise RuntimeError("No text encoder backend available. Install open_clip_torch and/or transformers and/or clip.")

    candidates = [b for b in backends if b.dim() == D]
    if len(candidates) == 0:
        dims = sorted(list(set([b.dim() for b in backends])))
        raise RuntimeError(f"No backend dim matches teacher D={D}. Available backend dims={dims}. "
                           f"Your teacher_pt text_feats dim is {D}; please ensure the matching encoder library is installed.")

    best_backend = candidates[0]
    best_score = -1e9

    # Evaluate each backend on sample captions
    for b in candidates:
        try:
            enc = b.encode(sample_captions, batch=32)  # [M, D]
            enc = F.normalize(enc.float(), dim=-1)
            # teacher_feats already normalized; pairwise per item
            # assume sample captions align to teacher_feats rows we passed in
            sims = (enc * teacher_feats[:enc.shape[0]].cpu()).sum(dim=-1)
            score = float(sims.mean().item())
            if score > best_score:
                best_score = score
                best_backend = b
        except Exception:
            continue

    # If score is very low, we still return best, but warn via print
    if best_score < 0.50:
        print(f"[WARN] Auto-selected backend '{best_backend.name}' but mean cosine to teacher is low ({best_score:.3f}). "
              f"This suggests teacher_feats may be produced by a different text encoder. Proceeding anyway.")

    print(f"[Encoder] Auto-selected backend: {best_backend.name} (mean cos={best_score:.4f}, D={D})")
    return best_backend

# ==========================================
# 3. Selection (teacher-space)
# ==========================================

def choose_best_hardneg(pos_feat_norm: torch.Tensor,
                        cand_feats_norm: torch.Tensor,
                        sim_low: float,
                        sim_high: float,
                        top_k: int = 1) -> List[int]:
    """
    Select Top-K hardest negatives by cosine similarity within [sim_low, sim_high].
    If not enough candidates fall in range, backfill with global top similarity.
    """
    sims = (cand_feats_norm @ pos_feat_norm.view(-1, 1)).squeeze(1)
    in_range = (sims >= sim_low) & (sims <= sim_high)
    idxs = torch.nonzero(in_range, as_tuple=False).view(-1)

    if idxs.numel() > 0:
        best_in_range = idxs[torch.argsort(sims[idxs], descending=True)].tolist()
    else:
        best_in_range = []

    final_idxs = list(best_in_range)
    if len(final_idxs) < top_k:
        for idx in torch.argsort(sims, descending=True).tolist():
            if idx not in final_idxs:
                final_idxs.append(idx)
            if len(final_idxs) >= top_k:
                break
    return [int(x) for x in final_idxs[:top_k]]

# ==========================================
# 4. Multi-thread API worker
# ==========================================

def process_item_api(img_id: int,
                     pos_cap: str,
                     max_retries: int,
                     sleep_base: float,
                     min_keep: int,
                     max_ratio_delta: float,
                     min_jaccard: float,
                     forbid_uncertain: bool) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    try:
        items = call_deepseek_json(
            pos_caption=pos_cap,
            max_retries=max_retries,
            sleep_base=sleep_base,
            min_keep=min_keep,
            max_ratio_delta=max_ratio_delta,
            min_jaccard=min_jaccard,
            forbid_uncertain=forbid_uncertain,
        )
        if not items:
            return img_id, None
        return img_id, items
    except Exception:
        return img_id, None

# ==========================================
# 5. Main
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_pt", type=str, default="data/nsd_text/train_coco_text_clip.pt")
    ap.add_argument("--caption_json", type=str, default="data/nsd_text/train_coco_captions.json")
    ap.add_argument("--out_pt", type=str, default="data/nsd_text/train_coco_captions_hard_negs_clip.pt")
    ap.add_argument("--out_jsonl", type=str, default="data/nsd_text/train_coco_captions_hard_negs.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)

    # IMPORTANT: training code currently supports K=1 only
    ap.add_argument("--top_k", type=int, default=1)

    ap.add_argument("--sim_low", type=float, default=0.15)
    ap.add_argument("--sim_high", type=float, default=0.85)

    # Multi-process sharding
    ap.add_argument("--shard_id", type=int, default=0, help="当前分片ID (0~num_shards-1)")
    ap.add_argument("--num_shards", type=int, default=1, help="总分片数")

    # API concurrency
    ap.add_argument("--workers", type=int, default=8, help="API并发请求数")

    # DeepSeek retry knobs
    ap.add_argument("--api_retries", type=int, default=6)
    ap.add_argument("--api_sleep_base", type=float, default=1.5)

    # Cleaning knobs
    ap.add_argument("--min_keep", type=int, default=6, help="clean后至少保留多少条候选，否则重试")
    ap.add_argument("--len_ratio", type=float, default=0.30)
    ap.add_argument("--min_jaccard", type=float, default=0.20)
    ap.add_argument("--forbid_uncertain", action="store_true", default=True)

    # Saving cadence
    ap.add_argument("--save_every", type=int, default=200)

    args = ap.parse_args()

    if args.top_k != 1:
        raise RuntimeError("Current training code expects hard-neg feats shape [N, D]. "
                           "Please use --top_k 1 (or modify training to support K>1).")

    if not os.path.isfile(args.teacher_pt):
        raise FileNotFoundError(f"teacher pt not found: {args.teacher_pt}")
    if not os.path.isfile(args.caption_json):
        raise FileNotFoundError(f"caption json not found: {args.caption_json}")

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    # Load teacher
    print(f"[Init] Loading teacher from: {args.teacher_pt}")
    teacher = torch.load(args.teacher_pt, map_location="cpu")
    if (not isinstance(teacher, dict)) or ("image_ids" not in teacher) or ("text_feats" not in teacher):
        raise RuntimeError(f"teacher_pt must be dict with keys: image_ids, text_feats. Got keys={list(teacher.keys()) if isinstance(teacher, dict) else type(teacher)}")

    image_ids: torch.Tensor = teacher["image_ids"].long().cpu()
    pos_text_feats: torch.Tensor = teacher["text_feats"].float().cpu()
    N = int(image_ids.numel())
    D = int(pos_text_feats.shape[-1])
    if pos_text_feats.ndim != 2 or pos_text_feats.shape[0] != N:
        raise RuntimeError(f"teacher text_feats shape mismatch: expected [N,D], got {tuple(pos_text_feats.shape)} with N={N}")

    # Load captions mapping (already prematched: image_ids aligned)
    print(f"[Init] Loading captions from: {args.caption_json}")
    mapping_data = json.loads(open(args.caption_json, "r", encoding="utf-8").read())
    if "image_ids" not in mapping_data or "captions" not in mapping_data:
        raise RuntimeError("caption_json must contain keys: image_ids, captions")

    cap_map: Dict[int, str] = {}
    for iid, cap in zip(mapping_data["image_ids"], mapping_data["captions"]):
        try:
            cap_map[int(iid)] = str(cap)
        except Exception:
            continue

    # Build id->row map
    img_id_to_idx = {int(iid.item()): i for i, iid in enumerate(image_ids)}

    # Resume done set (robust)
    done: set[int] = set()
    if args.resume and os.path.isfile(args.out_jsonl):
        with open(args.out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "image_id" in obj:
                        done.add(int(obj["image_id"]))
                except Exception:
                    pass
        print(f"[Resume] Found {len(done)} lines in existing jsonl.")

    # Prepare output tensors
    neg_text_feats = torch.zeros((N, D), dtype=torch.float32)
    valid_mask = torch.zeros((N,), dtype=torch.bool)

    # Load previous pt if resume
    if args.resume and os.path.isfile(args.out_pt):
        try:
            prev = torch.load(args.out_pt, map_location="cpu")
            if isinstance(prev, dict) and "image_ids" in prev and torch.equal(prev["image_ids"].long().cpu(), image_ids):
                if "neg_text_feats" in prev and isinstance(prev["neg_text_feats"], torch.Tensor) and prev["neg_text_feats"].shape == neg_text_feats.shape:
                    neg_text_feats = prev["neg_text_feats"].float().cpu()
                    print("[Resume] Loaded neg_text_feats from existing pt.")
                if "valid_mask" in prev and isinstance(prev["valid_mask"], torch.Tensor) and prev["valid_mask"].shape == valid_mask.shape:
                    valid_mask = prev["valid_mask"].bool().cpu()
                    print("[Resume] Loaded valid_mask from existing pt.")
        except Exception:
            pass

    # Build todo list (respect sharding + resume + limit)
    todo_list: List[Tuple[int, int, str]] = []
    total_target = N if args.limit <= 0 else min(N, int(args.limit))

    for idx in range(total_target):
        if args.num_shards > 1:
            if (idx % args.num_shards) != int(args.shard_id):
                continue

        img_id = int(image_ids[idx].item())
        if img_id in done:
            # if embedding already filled, skip
            if torch.norm(neg_text_feats[idx]).item() > 0:
                continue

        pos_cap = cap_map.get(img_id, "")
        if not pos_cap:
            continue

        todo_list.append((idx, img_id, pos_cap))

    print(f"[Todo] shard {args.shard_id}/{args.num_shards} -> {len(todo_list)} items (N={N}, D={D})")

    if len(todo_list) == 0:
        # Still save to ensure consistent file exists
        torch.save({"image_ids": image_ids, "neg_text_feats": neg_text_feats, "valid_mask": valid_mask}, args.out_pt)
        print(f"[Done] Nothing to do. Saved: {args.out_pt}")
        return

    # Auto-select encoder backend to match teacher space
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Sample a few captions that have teacher feats aligned (use the same indices as in todo_list)
    sample_m = min(32, len(todo_list))
    sample_idxs = [todo_list[i][0] for i in range(sample_m)]
    sample_caps = [todo_list[i][2] for i in range(sample_m)]
    teacher_sample = pos_text_feats[sample_idxs]  # [M, D]

    encoder_backend = auto_select_backend(
        teacher_feats=teacher_sample,
        sample_captions=sample_caps,
        device=device
    )

    # Open output jsonl (shard-suffixed if needed)
    out_jsonl_path = args.out_jsonl
    out_pt_path = args.out_pt
    if args.num_shards > 1:
        base, ext = os.path.splitext(out_jsonl_path)
        out_jsonl_path = f"{base}_part{args.shard_id}{ext or '.jsonl'}"
        basep, extp = os.path.splitext(out_pt_path)
        out_pt_path = f"{basep}_part{args.shard_id}{extp or '.pt'}"

    jf = open(out_jsonl_path, "a", encoding="utf-8")

    print(f"[Run] API workers={args.workers} | encoder='{encoder_backend.name}' | sim_window=[{args.sim_low},{args.sim_high}]")
    print(f"[Out] jsonl={out_jsonl_path} | pt={out_pt_path}")

    # Main loop: threadpool for API, main thread for GPU encode + selection + saving
    processed = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {}
        for idx, img_id, pos_cap in todo_list:
            fut = executor.submit(
                process_item_api,
                img_id, pos_cap,
                int(args.api_retries),
                float(args.api_sleep_base),
                int(args.min_keep),
                float(args.len_ratio),
                float(args.min_jaccard),
                bool(args.forbid_uncertain),
            )
            futures[fut] = (idx, img_id, pos_cap)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Generating(shard={args.shard_id})"):
            idx, img_id, pos_cap = futures[fut]
            _img_id, items = fut.result()
            if items is None or len(items) == 0:
                # Fallback to avoid zero vectors (training code doesn't mask invalid per-sample)
                # Use another random positive embedding as a negative embedding (still in teacher space).
                j = random.randrange(0, N)
                if j == idx:
                    j = (j + 1) % N
                neg_text_feats[idx] = F.normalize(pos_text_feats[j], dim=-1)
                valid_mask[idx] = False

                rec = {
                    "image_id": int(img_id),
                    "pos_caption": pos_cap,
                    "neg_caption": f"__FALLBACK_POS_EMB_FROM_IMAGE_ID__{int(image_ids[j].item())}",
                    "type": "fallback",
                    "edit": "",
                    "sim": float((neg_text_feats[idx] @ F.normalize(pos_text_feats[idx], dim=-1)).item()),
                    "fallback": True,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
            else:
                # Main-thread: encode candidates in teacher space
                cand_caps = [str(it.get("caption_cf", "")).strip() for it in items]
                # Remove empties just in case
                cand_caps = [c for c in cand_caps if c]
                if len(cand_caps) == 0:
                    # fallback again
                    j = random.randrange(0, N)
                    if j == idx:
                        j = (j + 1) % N
                    neg_text_feats[idx] = F.normalize(pos_text_feats[j], dim=-1)
                    valid_mask[idx] = False

                    rec = {
                        "image_id": int(img_id),
                        "pos_caption": pos_cap,
                        "neg_caption": f"__FALLBACK_POS_EMB_FROM_IMAGE_ID__{int(image_ids[j].item())}",
                        "type": "fallback",
                        "edit": "",
                        "sim": float((neg_text_feats[idx] @ F.normalize(pos_text_feats[idx], dim=-1)).item()),
                        "fallback": True,
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                else:
                    cand_feats = encoder_backend.encode(cand_caps, batch=32)  # [C, D] normalized
                    if cand_feats.shape[-1] != D:
                        raise RuntimeError(f"Dim mismatch: teacher D={D}, cand_feats D={cand_feats.shape[-1]} using backend={encoder_backend.name}")

                    pos_feat_i = F.normalize(pos_text_feats[idx].float(), dim=-1)
                    cand_feats_norm = F.normalize(cand_feats.float(), dim=-1)

                    idxs = choose_best_hardneg(
                        pos_feat_norm=pos_feat_i,
                        cand_feats_norm=cand_feats_norm,
                        sim_low=float(args.sim_low),
                        sim_high=float(args.sim_high),
                        top_k=1,
                    )
                    best = int(idxs[0])
                    neg_text_feats[idx] = cand_feats_norm[best].float().cpu()
                    valid_mask[idx] = True

                    chosen_item = items[best] if best < len(items) else {}
                    sim = float((cand_feats_norm[best].cpu() @ pos_feat_i.cpu()).item())

                    rec = {
                        "image_id": int(img_id),
                        "pos_caption": pos_cap,
                        "neg_caption": str(chosen_item.get("caption_cf", "")),
                        "type": str(chosen_item.get("type", "")),
                        "edit": str(chosen_item.get("edit", "")),
                        "changed_span": str(chosen_item.get("changed_span", "")),
                        "sim": sim,
                        "fallback": False,
                        "encoder_backend": encoder_backend.name,
                        "sim_window": [float(args.sim_low), float(args.sim_high)],
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()

            processed += 1
            if args.save_every > 0 and (processed % int(args.save_every) == 0):
                torch.save(
                    {"image_ids": image_ids, "neg_text_feats": neg_text_feats, "valid_mask": valid_mask},
                    out_pt_path
                )

    jf.close()

    # Final save
    torch.save({"image_ids": image_ids, "neg_text_feats": neg_text_feats, "valid_mask": valid_mask}, out_pt_path)
    print(f"[Done] Saved: {out_pt_path}")

if __name__ == "__main__":
    main()
