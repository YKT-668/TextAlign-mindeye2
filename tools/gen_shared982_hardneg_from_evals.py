#!/usr/bin/env python
# coding: utf-8
"""gen_shared982_hardneg_from_evals.py

Generate per-image counterfactual hard negative captions for the shared982 eval subset.

Requirements implemented:
- Positive captions: evals/all_captions.pt (aligned with exported *_ids.json order for 1000 shared images)
- ID order: any exported *_ids.json (len==1000)
- shared982 filtering: src/shared982.npy (bool mask over global COCO image ids)
- LLM generates counterfactual negatives with labeled type: object|attribute|relation
- Quality filtering (must):
  - forbid negation words: not/no/never/without
  - forbid uncertainty phrases: "this image", "looks like", "possibly", etc.
  - length within ±30% (word-count proxy)
  - minimal edit heuristic via token overlap (Jaccard)
  - OpenCLIP cosine similarity window on text embeddings (ViT-bigG-14 laion2b_s39b_b160k, 1280-d)
- Output:
  - cache/hardneg/shared982_hardneg.jsonl (K_final per image)
  - cache/hardneg/shared982_hardneg_meta.csv
  - cache/hardneg/shared982_hardneg_audit.json
  - cache/hardneg/shared982_hardneg_heartbeat.json
  - cache/hardneg/shared982_hardneg_for_ccd.jsonl (1 neg per image, for existing CCD reader)
  - optional embeddings: shared982_pos_text_emb.npy / shared982_neg_text_emb.npy

Notes:
- This script does NOT modify any existing LLM prompt code in the repo.
- API key is read from env var DEEPSEEK_API_KEY by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import urllib.request
import urllib.error


PROJ_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CAPTIONS_PT = PROJ_ROOT / "evals" / "all_captions.pt"
DEFAULT_SHARED982_MASK = PROJ_ROOT / "src" / "shared982.npy"
DEFAULT_IDS_JSON = PROJ_ROOT / "evals" / "brain_tokens" / "official_s1" / "subj01_ids.json"

DEFAULT_OUT_DIR = PROJ_ROOT / "cache" / "hardneg"
DEFAULT_OUT_JSONL = DEFAULT_OUT_DIR / "shared982_hardneg.jsonl"
DEFAULT_OUT_META = DEFAULT_OUT_DIR / "shared982_hardneg_meta.csv"
DEFAULT_OUT_AUDIT = DEFAULT_OUT_DIR / "shared982_hardneg_audit.json"
DEFAULT_HEARTBEAT = DEFAULT_OUT_DIR / "shared982_hardneg_heartbeat.json"

DEFAULT_POS_EMB = DEFAULT_OUT_DIR / "shared982_pos_text_emb.npy"
DEFAULT_NEG_EMB = DEFAULT_OUT_DIR / "shared982_neg_text_emb.npy"
DEFAULT_IMAGE_IDS_JSON = DEFAULT_OUT_DIR / "shared982_image_ids.json"

# For compatibility with existing eval_ccd_embed.py (expects one neg per image)
DEFAULT_OUT_JSONL_FOR_CCD = DEFAULT_OUT_DIR / "shared982_hardneg_for_ccd.jsonl"


NEGATION_RE = re.compile(r"\b(?:not|no|never|without)\b", flags=re.IGNORECASE)
UNCERTAIN_RE = re.compile(
	r"\b(?:this\s+image|looks\s+like|possibly|maybe|might|appears\s+to|seems\s+to|could\s+be)\b",
	flags=re.IGNORECASE,
)


def _now_ts() -> float:
	return float(time.time())


def _json_dumps(obj: Any) -> str:
	return json.dumps(obj, ensure_ascii=False)


def _safe_word_count(s: str) -> int:
	s = (s or "").strip()
	if not s:
		return 0
	return len([w for w in re.split(r"\s+", s) if w])


def _length_ok(pos: str, neg: str, max_ratio_delta: float) -> bool:
	p = max(1, _safe_word_count(pos))
	n = max(1, _safe_word_count(neg))
	ratio = n / p
	return (1.0 - max_ratio_delta) <= ratio <= (1.0 + max_ratio_delta)


def _token_overlap_ok(pos: str, neg: str, min_jaccard: float) -> bool:
	# Minimal-edit heuristic: require decent overlap.
	pos_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", pos)}
	neg_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", neg)}
	if not pos_words or not neg_words:
		return False
	inter = len(pos_words & neg_words)
	union = len(pos_words | neg_words)
	return (inter / max(1, union)) >= float(min_jaccard)


def _strip_quotes(s: str) -> str:
	s = (s or "").strip()
	if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
		return s[1:-1].strip()
	return s


def _load_captions(path: Path) -> List[str]:
	obj = torch.load(str(path), map_location="cpu")
	if isinstance(obj, np.ndarray):
		obj = obj.tolist()
	if isinstance(obj, list):
		return [str(x) for x in obj]
	raise RuntimeError(f"Unsupported captions container at {path}: {type(obj)}")


def _load_ids_json(path: Path) -> List[int]:
	ids = json.loads(path.read_text())
	if not isinstance(ids, list):
		raise RuntimeError(f"ids json must be a list, got {type(ids)}")
	return [int(x) for x in ids]


def _load_shared982_subset_ids(mask_path: Path) -> set[int]:
	m = np.load(str(mask_path))
	if m.dtype == np.bool_ and m.ndim == 1:
		idx = np.where(m > 0)[0].astype(np.int64)
		return {int(x) for x in idx.tolist()}
	if m.ndim == 1:
		return {int(x) for x in np.asarray(m, dtype=np.int64).tolist()}
	raise RuntimeError(f"shared982 mask has unexpected shape/dtype: {m.shape} {m.dtype}")


def _discover_ids_jsons() -> List[Path]:
	base = PROJ_ROOT / "evals" / "brain_tokens"
	if not base.is_dir():
		return []
	out: List[Path] = []
	out.extend(list(base.rglob("*_ids.json")))
	out.extend(list(base.rglob("ids.json")))
	return sorted(set(out))


def _auto_pick_ids_json(candidates: List[Path], expect_len: int = 1000) -> Optional[Path]:
	subj01 = [p for p in candidates if "subj01" in p.name]
	ordered = subj01 + [p for p in candidates if p not in subj01]
	for p in ordered:
		try:
			ids = _load_ids_json(p)
			if len(ids) == expect_len and len(set(ids)) == expect_len:
				return p
		except Exception:
			continue
	return None


def _normalize_base_url(base_url: str) -> str:
	u = (base_url or "").strip().rstrip("/")
	return u


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
	data = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(url, data=data, headers=headers, method="POST")
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		raw = resp.read().decode("utf-8")
	return json.loads(raw)


def _chat_completions(
	*,
	api_key: str,
	base_url: str,
	model: str,
	messages: List[Dict[str, str]],
	temperature: float,
	max_tokens: int,
	timeout: float,
	response_format: Optional[Dict[str, str]] = None,
) -> str:
	base = _normalize_base_url(base_url)
	# OpenAI-compatible path
	url = base + "/v1/chat/completions"
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}",
	}
	payload: Dict[str, Any] = {
		"model": model,
		"messages": messages,
		"temperature": float(temperature),
		"max_tokens": int(max_tokens),
		"stream": False,
	}
	if response_format:
		payload["response_format"] = response_format

	obj = _post_json(url, payload, headers=headers, timeout=float(timeout))
	# Expected: {choices:[{message:{content:"..."}}]}
	choices = obj.get("choices", [])
	if not choices:
		raise RuntimeError(f"LLM returned no choices: keys={list(obj.keys())}")
	msg = choices[0].get("message", {})
	content = msg.get("content", "")
	if not isinstance(content, str):
		content = str(content)
	return content


SYSTEM_PROMPT = (
	"You are an expert at generating hard negative captions for image-text contrastive learning.\n"
	"The user will provide an original English caption that correctly describes an image.\n"
	"Your task: generate counterfactual hard negative captions that are semantically close to the original "
	"but clearly WRONG as a description of that image.\n"
	"Requirements:\n"
	"1. Each negative must be plausible, fluent, and commonsense.\n"
	"2. Keep the same style and sentence structure as much as possible (minimal edit).\n"
	"3. Do NOT use negation words: not, no, never, without.\n"
	"4. Do NOT use uncertainty phrases like: this image, looks like, possibly, maybe, might, appears to, seems to, could be.\n"
	"5. Produce three types of edits: object, attribute, relation.\n"
	"6. Do NOT introduce a new scene; keep the same scene context.\n"
	"7. Output MUST be valid JSON, with no extra text.\n"
)


def _user_prompt(pos_caption: str, k_object: int, k_attribute: int, k_relation: int) -> str:
	return (
		"Original (positive) caption:\n"
		f"\"{pos_caption}\"\n\n"
		"Generate hard negative captions with minimal edits.\n"
		"Return JSON with a single key 'candidates' that is a list of objects.\n"
		"Each object must have keys: 'type' (object|attribute|relation) and 'caption' (string).\n"
		f"Counts: object={k_object}, attribute={k_attribute}, relation={k_relation}.\n"
		"Do NOT include any other keys.\n"
	)


@dataclass
class Candidate:
	image_id: int
	pos_caption: str
	neg_caption: str
	typ: str
	sim_text: float
	seed: int


@dataclass
class PerImageMeta:
	image_id: int
	idx_in_1000: int
	pos_len: int
	rounds_used: int
	window_low: float
	window_high: float
	raw_generated: int
	text_filter_kept: int
	sim_filter_kept: int
	final_kept: int
	final_object: int
	final_attribute: int
	final_relation: int
	fail_reason: str


def _encode_openclip_text(
	captions: Sequence[str],
	arch: str,
	pretrained: str,
	device: str,
	batch_size: int,
	progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
	model, tok = _get_openclip_model_and_tokenizer(arch=arch, pretrained=pretrained, device=device)

	outs: List[torch.Tensor] = []
	with torch.no_grad():
		total = int(len(captions))
		for i in range(0, len(captions), batch_size):
			batch = captions[i : i + batch_size]
			tokens = tok(list(batch)).to(device)
			z = model.encode_text(tokens)
			outs.append(z.float().detach().cpu())
			if progress_cb is not None:
				progress_cb(min(i + len(batch), total), total)
	return torch.cat(outs, dim=0)


_OPENCLIP_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


def _get_openclip_model_and_tokenizer(*, arch: str, pretrained: str, device: str):
	key = (str(arch), str(pretrained), str(device))
	obj = _OPENCLIP_CACHE.get(key)
	if obj is not None:
		return obj
	import open_clip

	model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
	model = model.to(device)
	model.eval()
	tok = open_clip.get_tokenizer(arch)
	_OPENCLIP_CACHE[key] = (model, tok)
	return model, tok


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a = torch.nn.functional.normalize(a, dim=-1)
	b = torch.nn.functional.normalize(b, dim=-1)
	return (a * b).sum(dim=-1)


def _load_done(out_jsonl: Path) -> Dict[int, int]:
	done: Dict[int, int] = {}
	if not out_jsonl.is_file():
		return done
	with out_jsonl.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
				iid = int(obj.get("image_id"))
				done[iid] = int(done.get(iid, 0)) + 1
			except Exception:
				continue
	return done


def _load_existing_candidates(out_jsonl: Path) -> Dict[int, List[Candidate]]:
	"""Load existing written candidates from jsonl for resume.

	We only need fields that we write: image_id, pos_caption, neg_caption, type, sim_text, seed.
	"""
	out: Dict[int, List[Candidate]] = {}
	if not out_jsonl.is_file():
		return out
	with out_jsonl.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
				iid = int(obj.get("image_id"))
				pos = str(obj.get("pos_caption", ""))
				neg = str(obj.get("neg_caption", ""))
				typ = str(obj.get("type", "")).strip().lower()
				sim = float(obj.get("sim_text", 0.0))
				seed = int(obj.get("seed", 0))
				if iid and pos and neg and typ:
					out.setdefault(iid, []).append(Candidate(iid, pos, neg, typ, sim, seed))
			except Exception:
				continue
	return out


def _write_heartbeat(path: Path, payload: Dict[str, Any]) -> None:
	tmp = path.with_suffix(".tmp")
	tmp.write_text(_json_dumps(payload))
	tmp.replace(path)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--captions_pt", type=str, default=str(DEFAULT_CAPTIONS_PT))
	ap.add_argument("--ids_json", type=str, default="")
	ap.add_argument("--shared982_mask", type=str, default=str(DEFAULT_SHARED982_MASK))

	ap.add_argument("--out_jsonl", type=str, default=str(DEFAULT_OUT_JSONL))
	ap.add_argument("--out_meta", type=str, default=str(DEFAULT_OUT_META))
	ap.add_argument("--out_audit", type=str, default=str(DEFAULT_OUT_AUDIT))
	ap.add_argument("--out_jsonl_for_ccd", type=str, default=str(DEFAULT_OUT_JSONL_FOR_CCD))

	ap.add_argument("--save_embeddings", action="store_true")
	ap.add_argument("--pos_emb_npy", type=str, default=str(DEFAULT_POS_EMB))
	ap.add_argument("--neg_emb_npy", type=str, default=str(DEFAULT_NEG_EMB))
	ap.add_argument("--image_ids_out", type=str, default=str(DEFAULT_IMAGE_IDS_JSON))

	ap.add_argument("--clip_model", type=str, default="ViT-bigG-14")
	ap.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
	_default_clip_device = "cuda" if torch.cuda.is_available() else "cpu"
	ap.add_argument("--clip_device", type=str, default=_default_clip_device, choices=["cpu", "cuda"])
	ap.add_argument("--clip_batch", type=int, default=128)
	ap.add_argument("--neg_encode_batch_target", type=int, default=2048)

	ap.add_argument("--k_raw", type=int, default=12)
	ap.add_argument("--k_final", type=int, default=8)
	ap.add_argument("--low", type=float, default=0.25)
	ap.add_argument("--high", type=float, default=0.55)
	ap.add_argument("--max_rounds", type=int, default=5)
	ap.add_argument("--window_step", type=float, default=0.05)
	ap.add_argument("--min_low", type=float, default=0.15)
	ap.add_argument("--max_high", type=float, default=0.75)

	ap.add_argument("--max_len_ratio_delta", type=float, default=0.30)
	ap.add_argument("--min_jaccard", type=float, default=0.45)

	ap.add_argument("--workers", type=int, default=4)
	ap.add_argument("--seed", type=int, default=42)

	ap.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
	ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
	ap.add_argument("--model", type=str, default="deepseek-chat")
	ap.add_argument("--timeout", type=float, default=30.0)
	ap.add_argument("--max_retries", type=int, default=3)
	ap.add_argument("--temperature", type=float, default=0.7)
	ap.add_argument("--max_tokens", type=int, default=500)

	ap.add_argument("--heartbeat_path", type=str, default=str(DEFAULT_HEARTBEAT))
	ap.add_argument("--heartbeat_interval_sec", type=float, default=60.0)

	args = ap.parse_args()

	captions_pt = Path(args.captions_pt).resolve()
	shared982_mask = Path(args.shared982_mask).resolve()

	out_jsonl = Path(args.out_jsonl).resolve()
	out_meta = Path(args.out_meta).resolve()
	out_audit = Path(args.out_audit).resolve()
	out_jsonl_for_ccd = Path(args.out_jsonl_for_ccd).resolve()

	out_dir = out_jsonl.parent
	out_dir.mkdir(parents=True, exist_ok=True)

	if not captions_pt.is_file():
		raise FileNotFoundError(f"Missing captions_pt: {captions_pt}")
	if not shared982_mask.is_file():
		raise FileNotFoundError(f"Missing shared982 mask: {shared982_mask}")

	# ids_json resolution
	if args.ids_json.strip():
		ids_json_path = Path(args.ids_json).resolve()
	else:
		candidates = _discover_ids_jsons()
		ids_json_path = _auto_pick_ids_json(candidates, expect_len=1000) or (
			DEFAULT_IDS_JSON if DEFAULT_IDS_JSON.is_file() else None
		)
	if ids_json_path is None or not Path(ids_json_path).is_file():
		raise RuntimeError("Could not find a valid *_ids.json (len=1000). Provide --ids_json explicitly.")
	ids_json_path = Path(ids_json_path).resolve()

	captions_1000 = _load_captions(captions_pt)
	ids_1000 = _load_ids_json(ids_json_path)
	if len(captions_1000) != len(ids_1000):
		raise RuntimeError(
			f"Alignment error: captions len={len(captions_1000)} != ids len={len(ids_1000)}. "
			"captions_pt must align with ids_json order."
		)
	if len(captions_1000) != 1000:
		raise RuntimeError(f"Expected 1000 shared captions/ids, got {len(captions_1000)}")

	subset_ids = _load_shared982_subset_ids(shared982_mask)
	keep_idx: List[int] = [i for i, iid in enumerate(ids_1000) if int(iid) in subset_ids]
	if len(keep_idx) != 982:
		raise RuntimeError(f"shared982 filter mismatch: keep={len(keep_idx)} (expected 982)")

	image_ids = [int(ids_1000[i]) for i in keep_idx]
	pos_captions = [str(captions_1000[i]) for i in keep_idx]

	Path(args.image_ids_out).resolve().write_text(_json_dumps(image_ids))

	done_counts = _load_done(out_jsonl)
	pending: List[int] = [
		j for j, iid in enumerate(image_ids) if int(done_counts.get(int(iid), 0)) < int(args.k_final)
	]

	if not args.api_key:
		raise RuntimeError("DEEPSEEK_API_KEY is missing. Set env var DEEPSEEK_API_KEY or pass --api_key.")

	t0 = _now_ts()
	print(f"[INFO] ids_json={ids_json_path}")
	print(f"[INFO] captions_pt={captions_pt} (len=1000)")
	print(f"[INFO] shared982 keep={len(image_ids)} pending={len(pending)}")
	print(f"[INFO] Encoding pos captions with OpenCLIP {args.clip_model}/{args.clip_pretrained} on {args.clip_device} ...")

	hb_path = Path(args.heartbeat_path).resolve()
	hb_lock = threading.Lock()
	hb_stop = threading.Event()
	hb_state: Dict[str, Any] = {
		"stage": "startup",
		"round": 0,
		"done_images": 0,
		"total_images": 982,
		"pending_images": int(len(pending)),
		"encoded": 0,
		"total": 0,
	}

	def _heartbeat_worker() -> None:
		while not hb_stop.is_set():
			now = _now_ts()
			with hb_lock:
				payload = {
					"ts": now,
					"elapsed_sec": now - t0,
					"stage": hb_state.get("stage"),
					"round": int(hb_state.get("round", 0)),
					"done_images": int(hb_state.get("done_images", 0)),
					"total_images": int(hb_state.get("total_images", 982)),
					"pending_images": int(hb_state.get("pending_images", 0)),
				}
				if "encoded" in hb_state and "total" in hb_state:
					payload["encoded"] = int(hb_state.get("encoded", 0))
					payload["total"] = int(hb_state.get("total", 0))
			_write_heartbeat(hb_path, payload)
			hb_stop.wait(float(args.heartbeat_interval_sec))

	threading.Thread(target=_heartbeat_worker, daemon=True).start()

	def maybe_stage_heartbeat(stage: str, encoded: int, total: int) -> None:
		with hb_lock:
			hb_state["stage"] = str(stage)
			hb_state["round"] = 0
			hb_state["done_images"] = 0
			hb_state["pending_images"] = int(len(pending))
			hb_state["encoded"] = int(encoded)
			hb_state["total"] = int(total)

	pos_emb_path = Path(args.pos_emb_npy).resolve()
	pos_emb: torch.Tensor
	if pos_emb_path.is_file():
		try:
			arr = np.load(str(pos_emb_path))
			if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 982:
				pos_emb = torch.from_numpy(arr.astype(np.float32))
				print(f"[INFO] Loaded cached pos_emb from {pos_emb_path}")
			else:
				raise RuntimeError("bad shape")
		except Exception:
			pos_emb_path.unlink(missing_ok=True)
			pos_emb = None  # type: ignore
	else:
		pos_emb = None  # type: ignore

	if pos_emb is None:
		maybe_stage_heartbeat("pos_encode", 0, len(pos_captions))
		pos_emb = _encode_openclip_text(
			pos_captions,
			arch=args.clip_model,
			pretrained=args.clip_pretrained,
			device=args.clip_device,
			batch_size=int(args.clip_batch),
			progress_cb=lambda enc, tot: maybe_stage_heartbeat("pos_encode", enc, tot),
		)
		# Always cache pos_emb for fast restart
		np.save(str(pos_emb_path), pos_emb.numpy().astype(np.float32))
		print(f"[INFO] Saved cached pos_emb to {pos_emb_path}")

	# Pre-normalize pos embeddings for fast cosine with neg embeddings.
	pos_emb = torch.nn.functional.normalize(pos_emb.float(), dim=-1).cpu()

	rng = np.random.default_rng(int(args.seed))

	kept: Dict[int, List[Candidate]] = {int(iid): [] for iid in image_ids}
	meta: Dict[int, PerImageMeta] = {}

	# Resume from existing outputs (streaming append mode)
	existing = _load_existing_candidates(out_jsonl)
	for iid, cands in existing.items():
		if int(iid) in kept:
			kept[int(iid)].extend(cands)

	# Track which images already have >=k_final written to disk
	written_images = {int(iid) for iid, n in done_counts.items() if int(n) >= int(args.k_final)}

	# Stream output files (append). We'll only append for images not yet complete.
	def select_final_for_image(cands: List[Candidate]) -> Tuple[List[Candidate], Dict[str, int]]:
		if not cands:
			return [], {"object": 0, "attribute": 0, "relation": 0}
		sims = np.array([c.sim_text for c in cands], dtype=np.float32)
		mid = float((float(args.low) + float(args.high)) / 2.0)
		order = np.argsort(np.abs(sims - mid))
		ordered = [cands[i] for i in order.tolist()]
		k = int(args.k_final)
		target_per = max(1, k // 3)
		out: List[Candidate] = []
		counts = {"object": 0, "attribute": 0, "relation": 0}
		for c in ordered:
			if len(out) >= k:
				break
			if counts.get(c.typ, 0) >= target_per:
				continue
			out.append(c)
			counts[c.typ] = counts.get(c.typ, 0) + 1
		if len(out) < k:
			for c in ordered:
				if len(out) >= k:
					break
				if c in out:
					continue
				out.append(c)
				counts[c.typ] = counts.get(c.typ, 0) + 1
		return out, counts

	def append_image_outputs(iid: int, selected: List[Candidate]) -> None:
		# Append K_final lines for this image to main jsonl, and 1 line to CCD jsonl.
		# We rely on done_counts + written_images to avoid duplicates.
		out_jsonl.parent.mkdir(parents=True, exist_ok=True)
		with out_jsonl.open("a", encoding="utf-8") as f:
			for c in selected:
				rec = {
					"image_id": int(c.image_id),
					"pos_caption": c.pos_caption,
					"neg_caption": c.neg_caption,
					"type": c.typ,
					"sim_text": float(c.sim_text),
					"seed": int(c.seed),
				}
				f.write(_json_dumps(rec) + "\n")
		# One neg per image for CCD
		best = max(selected, key=lambda x: x.sim_text)
		with out_jsonl_for_ccd.open("a", encoding="utf-8") as f:
			rec = {
				"image_id": int(best.image_id),
				"pos_caption": best.pos_caption,
				"neg_caption": best.neg_caption,
				"type": best.typ,
				"sim_text": float(best.sim_text),
				"seed": int(best.seed),
			}
			f.write(_json_dumps(rec) + "\n")

	audit_counts = {
		"final_total": 0,
		"final_by_type": {"object": 0, "attribute": 0, "relation": 0},
		"text_filter_reject_negation": 0,
		"text_filter_reject_uncertain": 0,
		"text_filter_reject_length": 0,
		"text_filter_reject_overlap": 0,
		"text_filter_reject_bad_type": 0,
		"parse_fail": 0,
	}

	def maybe_heartbeat(done_n: int, round_id: int) -> None:
		with hb_lock:
			hb_state["stage"] = "generate"
			hb_state["round"] = int(round_id)
			hb_state["done_images"] = int(done_n)
			hb_state["pending_images"] = int(sum(1 for iid in image_ids if len(kept[int(iid)]) < int(args.k_final)))
			hb_state["encoded"] = 0
			hb_state["total"] = 0

	def text_filter(pos: str, typ: str, neg: str) -> Tuple[bool, str]:
		typ = (typ or "").strip().lower()
		if typ not in ("object", "attribute", "relation"):
			return False, "bad_type"
		neg2 = _strip_quotes(neg)
		if not neg2:
			return False, "empty"
		if NEGATION_RE.search(neg2):
			return False, "negation"
		if UNCERTAIN_RE.search(neg2):
			return False, "uncertain"
		if not _length_ok(pos, neg2, float(args.max_len_ratio_delta)):
			return False, "length"
		if not _token_overlap_ok(pos, neg2, float(args.min_jaccard)):
			return False, "overlap"
		return True, "ok"

	def request_candidates(pos_caption: str, k_raw: int, call_seed: int) -> Optional[List[Dict[str, str]]]:
		k_obj = k_raw // 3
		k_attr = k_raw // 3
		k_rel = k_raw // 3
		rem = k_raw - (k_obj + k_attr + k_rel)
		if rem >= 1:
			k_obj += 1
		if rem >= 2:
			k_attr += 1
		user_msg = _user_prompt(pos_caption, k_obj, k_attr, k_rel)
		for attempt in range(1, int(args.max_retries) + 1):
			try:
				text = _chat_completions(
					api_key=str(args.api_key),
					base_url=str(args.base_url),
					model=str(args.model),
					messages=[
						{"role": "system", "content": SYSTEM_PROMPT},
						{"role": "user", "content": user_msg},
					],
					temperature=float(args.temperature),
					max_tokens=int(args.max_tokens),
					timeout=float(args.timeout),
					response_format={"type": "json_object"},
				)
				obj = json.loads(text)
				cands = obj.get("candidates", None)
				if not isinstance(cands, list):
					return None
				out: List[Dict[str, str]] = []
				for c in cands:
					if not isinstance(c, dict):
						continue
					typ = str(c.get("type", "")).strip().lower()
					cap = str(c.get("caption", "")).strip()
					if typ and cap:
						out.append({"type": typ, "caption": cap})
				return out
			except Exception:
				if attempt >= int(args.max_retries):
					return None
				time.sleep(0.6 * attempt)
		return None

	# No per-thread client objects needed (urllib is stateless).

	low = float(args.low)
	high = float(args.high)

	def done_images_count() -> int:
		return sum(1 for iid in image_ids if len(kept[int(iid)]) >= int(args.k_final))

	for round_id in range(1, int(args.max_rounds) + 1):
		need_idx = [j for j in pending if len(kept[int(image_ids[j])]) < int(args.k_final)]
		if not need_idx:
			break
		maybe_heartbeat(done_images_count(), round_id)
		print(f"[ROUND {round_id}] need={len(need_idx)} low={low:.2f} high={high:.2f} k_raw={int(args.k_raw)} k_final={int(args.k_final)}")

		# Accumulate candidates across images and batch-encode to speed up OpenCLIP.
		pending_caps: List[str] = []
		pending_typ: List[str] = []
		pending_iid: List[int] = []
		pending_posidx: List[int] = []
		pending_seed: List[int] = []

		def flush_pending() -> None:
			nonlocal pending_caps, pending_typ, pending_iid, pending_posidx, pending_seed
			if not pending_caps:
				return
			neg_emb = _encode_openclip_text(
				pending_caps,
				arch=args.clip_model,
				pretrained=args.clip_pretrained,
				device=args.clip_device,
				batch_size=int(args.clip_batch),
			)
			neg_emb = torch.nn.functional.normalize(neg_emb.float(), dim=-1).cpu()
			pos_e = pos_emb[torch.tensor(pending_posidx, dtype=torch.long)]
			sims = (pos_e * neg_emb).sum(dim=-1).numpy().astype(np.float32)

			for cap, typ, iid, pos_j, seed, sim in zip(pending_caps, pending_typ, pending_iid, pending_posidx, pending_seed, sims.tolist()):
				if float(sim) < low or float(sim) > high:
					continue
				# pos caption is the shared982 positive caption for this image
				pos_cap = pos_captions[pos_j]
				kept[int(iid)].append(Candidate(int(iid), pos_cap, cap, typ, float(sim), int(seed)))
				# If enough, stream write immediately
				if int(iid) not in written_images and len(kept[int(iid)]) >= int(args.k_final):
					selected, counts = select_final_for_image(kept[int(iid)])
					if len(selected) >= int(args.k_final):
						append_image_outputs(int(iid), selected[: int(args.k_final)])
						written_images.add(int(iid))
						kept[int(iid)] = selected[: int(args.k_final)]
						m = meta.get(int(iid))
						if m is None:
							m = PerImageMeta(
								image_id=int(iid),
								idx_in_1000=int(keep_idx[pos_j]),
								pos_len=_safe_word_count(pos_cap),
								rounds_used=round_id,
								window_low=low,
								window_high=high,
								raw_generated=0,
								text_filter_kept=0,
								sim_filter_kept=0,
								final_kept=0,
								final_object=0,
								final_attribute=0,
								final_relation=0,
								fail_reason="",
							)
						m.final_kept = int(args.k_final)
						m.final_object = int(counts.get("object", 0))
						m.final_attribute = int(counts.get("attribute", 0))
						m.final_relation = int(counts.get("relation", 0))
						meta[int(iid)] = m

			# reset
			pending_caps = []
			pending_typ = []
			pending_iid = []
			pending_posidx = []
			pending_seed = []

		futures = {}
		with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
			for t, j in enumerate(need_idx):
				iid = int(image_ids[j])
				pos = pos_captions[j]
				call_seed = int(rng.integers(0, 2**31 - 1))
				fut = ex.submit(request_candidates, pos, int(args.k_raw), call_seed)
				futures[fut] = (j, iid, call_seed)

			for fut in as_completed(futures):
				# Keep heartbeat fresh even when LLM calls are slow.
				maybe_heartbeat(done_images_count(), round_id)
				j, iid, call_seed = futures[fut]
				pos = pos_captions[j]
				res = fut.result()
				if res is None:
					audit_counts["parse_fail"] += 1
					if iid not in meta:
						meta[iid] = PerImageMeta(
							image_id=iid,
							idx_in_1000=int(keep_idx[j]),
							pos_len=_safe_word_count(pos),
							rounds_used=round_id,
							window_low=low,
							window_high=high,
							raw_generated=0,
							text_filter_kept=0,
							sim_filter_kept=0,
							final_kept=len(kept[iid]),
							final_object=sum(1 for c in kept[iid] if c.typ == "object"),
							final_attribute=sum(1 for c in kept[iid] if c.typ == "attribute"),
							final_relation=sum(1 for c in kept[iid] if c.typ == "relation"),
							fail_reason="llm_parse_fail",
						)
					continue

				raw_n = len(res)
				text_kept: List[Tuple[str, str]] = []
				rej = {"bad_type": 0, "negation": 0, "uncertain": 0, "length": 0, "overlap": 0, "empty": 0}

				for row in res:
					typ = str(row.get("type", "")).strip().lower()
					neg = str(row.get("caption", "")).strip()
					ok, reason = text_filter(pos, typ, neg)
					if not ok:
						if reason in rej:
							rej[reason] += 1
						continue
					text_kept.append((typ, _strip_quotes(neg)))

				audit_counts["text_filter_reject_bad_type"] += rej["bad_type"]
				audit_counts["text_filter_reject_negation"] += rej["negation"]
				audit_counts["text_filter_reject_uncertain"] += rej["uncertain"]
				audit_counts["text_filter_reject_length"] += rej["length"]
				audit_counts["text_filter_reject_overlap"] += rej["overlap"]

				seen = {c.neg_caption.lower() for c in kept[iid]}
				uniq: List[Tuple[str, str]] = []
				for typ, cap in text_kept:
					key = cap.lower()
					if key in seen:
						continue
					seen.add(key)
					uniq.append((typ, cap))

				if uniq:
					for typ, cap in uniq:
						pending_caps.append(cap)
						pending_typ.append(typ)
						pending_iid.append(int(iid))
						pending_posidx.append(int(j))
						pending_seed.append(int(call_seed))
					if len(pending_caps) >= int(args.neg_encode_batch_target):
						flush_pending()

				prev = meta.get(iid)
				meta[iid] = PerImageMeta(
					image_id=iid,
					idx_in_1000=int(keep_idx[j]),
					pos_len=_safe_word_count(pos),
					rounds_used=round_id,
					window_low=low,
					window_high=high,
					raw_generated=(prev.raw_generated if prev else 0) + raw_n,
					text_filter_kept=(prev.text_filter_kept if prev else 0) + len(uniq),
					sim_filter_kept=len(kept[iid]),
					final_kept=min(len(kept[iid]), int(args.k_final)),
					final_object=0,
					final_attribute=0,
					final_relation=0,
					fail_reason="",
				)

			# Flush remaining pending captions for this round.
			flush_pending()

		low = max(float(args.min_low), low - float(args.window_step))
		high = min(float(args.max_high), high + float(args.window_step))

	# Finalization: append any missing images that reached k_final but weren't written yet
	for iid in image_ids:
		iid = int(iid)
		if iid in written_images:
			continue
		if len(kept[iid]) >= int(args.k_final):
			selected, counts = select_final_for_image(kept[iid])
			if len(selected) >= int(args.k_final):
				append_image_outputs(iid, selected[: int(args.k_final)])
				written_images.add(iid)
				kept[iid] = selected[: int(args.k_final)]
				m = meta.get(iid)
				if m is None:
					j = image_ids.index(iid)
					m = PerImageMeta(iid, int(keep_idx[j]), _safe_word_count(pos_captions[j]), int(args.max_rounds), float(args.low), float(args.high), 0, 0, len(kept[iid]), int(args.k_final), 0, 0, 0, "")
				m.final_kept = int(args.k_final)
				m.final_object = int(counts.get("object", 0))
				m.final_attribute = int(counts.get("attribute", 0))
				m.final_relation = int(counts.get("relation", 0))
				meta[iid] = m

	# For images still missing, mark meta fail reason
	for j, iid in enumerate(image_ids):
		iid = int(iid)
		if iid not in meta:
			meta[iid] = PerImageMeta(iid, int(keep_idx[j]), _safe_word_count(pos_captions[j]), int(args.max_rounds), float(args.low), float(args.high), 0, 0, 0, 0, 0, 0, 0, "no_candidates_after_filters")
		if iid not in written_images:
			meta[iid].fail_reason = (meta[iid].fail_reason or "") + "|not_written"

	# Meta CSV
	out_meta_tmp = out_meta.with_suffix(".csv.tmp")
	with out_meta_tmp.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow([
			"image_id",
			"idx_in_1000",
			"pos_len_words",
			"rounds_used",
			"window_low",
			"window_high",
			"raw_generated",
			"text_filter_kept",
			"sim_filter_kept",
			"final_kept",
			"final_object",
			"final_attribute",
			"final_relation",
			"fail_reason",
		])
		for iid in image_ids:
			m = meta[int(iid)]
			w.writerow([
				m.image_id,
				m.idx_in_1000,
				m.pos_len,
				m.rounds_used,
				f"{m.window_low:.4f}",
				f"{m.window_high:.4f}",
				m.raw_generated,
				m.text_filter_kept,
				m.sim_filter_kept,
				m.final_kept,
				m.final_object,
				m.final_attribute,
				m.final_relation,
				m.fail_reason,
			])
	out_meta_tmp.replace(out_meta)

	# Optional embeddings for final negs
	if args.save_embeddings:
		k = int(args.k_final)
		neg_mat = np.zeros((982, k, int(pos_emb.shape[1])), dtype=np.float32)
		for j, iid in enumerate(image_ids):
			cands = kept[int(iid)][:k]
			if not cands:
				continue
			neg_caps = [c.neg_caption for c in cands]
			neg_e = _encode_openclip_text(
				neg_caps,
				arch=args.clip_model,
				pretrained=args.clip_pretrained,
				device=args.clip_device,
				batch_size=min(int(args.clip_batch), max(1, len(neg_caps))),
			).numpy().astype(np.float32)
			neg_mat[j, : neg_e.shape[0], :] = neg_e
		np.save(str(Path(args.neg_emb_npy).resolve()), neg_mat)

	# Audit summary
	sims_all: List[float] = []
	len_ratio_all: List[float] = []
	negation_hits = 0
	type_counts = {"object": 0, "attribute": 0, "relation": 0}
	for j, iid in enumerate(image_ids):
		pos = pos_captions[j]
		p_len = max(1, _safe_word_count(pos))
		for c in kept[int(iid)][: int(args.k_final)]:
			type_counts[c.typ] = type_counts.get(c.typ, 0) + 1
			sims_all.append(float(c.sim_text))
			n_len = max(1, _safe_word_count(c.neg_caption))
			len_ratio_all.append(float(n_len / p_len))
			if NEGATION_RE.search(c.neg_caption):
				negation_hits += 1

	audit = {
		"inputs": {
			"captions_pt": str(captions_pt),
			"ids_json": str(ids_json_path),
			"shared982_mask": str(shared982_mask),
			"n_shared1000": 1000,
			"n_shared982": 982,
		},
		"params": {
			"k_raw": int(args.k_raw),
			"k_final": int(args.k_final),
			"sim_low_init": float(args.low),
			"sim_high_init": float(args.high),
			"max_rounds": int(args.max_rounds),
			"clip_model": str(args.clip_model),
			"clip_pretrained": str(args.clip_pretrained),
			"clip_device": str(args.clip_device),
		},
		"final": {
			"total_negs": int(sum(type_counts.values())),
			"type_counts": type_counts,
			"negation_rate": float(negation_hits / max(1, sum(type_counts.values()))),
			"sim_text": {
				"mean": float(np.mean(sims_all)) if sims_all else float("nan"),
				"std": float(np.std(sims_all)) if sims_all else float("nan"),
				"min": float(np.min(sims_all)) if sims_all else float("nan"),
				"p05": float(np.quantile(sims_all, 0.05)) if sims_all else float("nan"),
				"p50": float(np.quantile(sims_all, 0.50)) if sims_all else float("nan"),
				"p95": float(np.quantile(sims_all, 0.95)) if sims_all else float("nan"),
				"max": float(np.max(sims_all)) if sims_all else float("nan"),
			},
			"len_ratio": {
				"mean": float(np.mean(len_ratio_all)) if len_ratio_all else float("nan"),
				"std": float(np.std(len_ratio_all)) if len_ratio_all else float("nan"),
				"min": float(np.min(len_ratio_all)) if len_ratio_all else float("nan"),
				"p05": float(np.quantile(len_ratio_all, 0.05)) if len_ratio_all else float("nan"),
				"p50": float(np.quantile(len_ratio_all, 0.50)) if len_ratio_all else float("nan"),
				"p95": float(np.quantile(len_ratio_all, 0.95)) if len_ratio_all else float("nan"),
				"max": float(np.max(len_ratio_all)) if len_ratio_all else float("nan"),
			},
		},
		"filters": audit_counts,
		"outputs": {
			"jsonl": str(out_jsonl),
			"jsonl_for_ccd": str(out_jsonl_for_ccd),
			"meta_csv": str(out_meta),
			"audit_json": str(out_audit),
			"pos_emb_npy": str(Path(args.pos_emb_npy).resolve()) if args.save_embeddings else None,
			"neg_emb_npy": str(Path(args.neg_emb_npy).resolve()) if args.save_embeddings else None,
			"image_ids_json": str(Path(args.image_ids_out).resolve()),
			"heartbeat": str(hb_path),
		},
	}
	out_audit.write_text(_json_dumps(audit))

	maybe_heartbeat(done_images_count(), int(args.max_rounds))
	with hb_lock:
		hb_state["stage"] = "done"
	hb_stop.set()

	dt = _now_ts() - t0
	n_ok = sum(1 for iid in image_ids if meta[int(iid)].final_kept >= int(args.k_final))
	print(f"[DONE] wrote {out_jsonl}")
	print(f"[DONE] wrote {out_meta}")
	print(f"[DONE] wrote {out_audit}")
	print(f"[DONE] wrote {out_jsonl_for_ccd}")
	print(f"[SUMMARY] ok_images={n_ok}/982 runtime_sec={dt:.1f}")


if __name__ == "__main__":
	main()