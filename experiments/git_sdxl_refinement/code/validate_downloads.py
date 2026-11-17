#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConceptAlign — 下载完成后的数据完整性 + ID 对齐检查。

检查项（详见任务第九节）：
  1. 必需文件是否全部存在
  2. 实际大小是否与远端元数据一致
  3. SHA256 是否匹配
  4. Ours 三个文件是否来自同一个正式 run
  5. 是否为正式 Ours S1 输出（核对 provenance/checksum）
  6. 重建图像 / GIT Caption / ID 样本数量是否一致
  7. 是否有重复或缺失 ID
  8. GT 图像 / GT Caption 与 Ours ID 是否可正确对齐
  9. 空 Caption、异常数据、非有限数值
  10. SDXL 权重是否为完整 safetensors

生成 reports/：
  DOWNLOAD_FINAL_REPORT.md  ASSET_INTEGRITY_REPORT.md
  ASSET_ALIGNMENT_REPORT.md MODEL_PROVENANCE.md
最终状态只可能是 READY_FOR_GPU 或 BLOCKED_MISSING_REQUIRED_ASSETS。
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"))
MANIFEST = ROOT / "manifests" / "download_manifest.tsv"
REPORTS = ROOT / "reports"
PROVENANCE_DIR = REPORTS / "provenance"

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except Exception:
    safe_open = None
    HAS_SAFETENSORS = False


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_bytes(n) -> str:
    n = float(n or 0)
    if n == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{int(n)} B" if units[i] == "B" else f"{n:.2f} {units[i]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list:
    entries = []
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.strip() for c in cols]
                continue
            entries.append({header[i]: (cols[i] if i < len(cols) else "") for i in range(len(header))})
    return entries


def write_report(name: str, content: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / name
    p.write_text(content, encoding="utf-8")
    return p


def describe_brief(obj):
    if torch is not None and torch.is_tensor(obj):
        return f"tensor shape={list(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, (list, tuple)):
        return f"list len={len(obj)}"
    if isinstance(obj, dict):
        return f"dict keys={list(obj.keys())[:10]}"
    return type(obj).__name__


def load_pt(path: Path, mmap: bool = False):
    """返回 (obj, error)。"""
    if not HAS_TORCH:
        return None, "torch 未安装"
    try:
        if mmap:
            obj = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        else:
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return obj, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def n_samples_of(obj) -> int:
    """返回对象的主样本数（tensor 首维 / list/dict 长度）。"""
    if torch is not None and torch.is_tensor(obj):
        return int(obj.shape[0]) if obj.ndim > 0 else 1
    if isinstance(obj, (list, tuple)):
        return len(obj)
    if isinstance(obj, dict):
        # 取第一个 tensor 或 list 的长度
        for v in obj.values():
            n = n_samples_of(v)
            if n is not None:
                return n
    return 0


def extract_ids(obj):
    """从 .pt 对象中提取一维整数 ID。返回 (list[int], error)。"""
    if obj is None:
        return None, "empty object"
    if torch is not None and torch.is_tensor(obj):
        t = obj.flatten().detach().cpu()
        return [int(x) for x in t.tolist()], None
    if isinstance(obj, (list, tuple)):
        return [int(x) for x in obj], None
    if isinstance(obj, dict):
        for v in obj.values():
            ids, err = extract_ids(v)
            if err is None:
                return ids, None
        return None, "no ID-like tensor found in dict"
    return None, f"unsupported object type {type(obj).__name__}"


def extract_captions(obj):
    """从 .pt 对象中提取字符串 caption 列表。返回 (list[str], error)。"""
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj], None
    if isinstance(obj, dict):
        for v in obj.values():
            caps, err = extract_captions(v)
            if err is None:
                return caps, None
        return None, "no caption-like list found in dict"
    if torch is not None and torch.is_tensor(obj) and obj.ndim <= 1:
        return [str(x) for x in obj.tolist()], None
    return None, f"unsupported caption object type {type(obj).__name__}"


def check_nonfinite(obj) -> tuple:
    """检查 tensor 中的非有限值。返回 (has_nonfinite, has_nan, has_inf)。"""
    if torch is None or not torch.is_tensor(obj):
        return False, False, False
    if obj.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return False, False, False
    try:
        flat = obj.detach().flatten()
        has_nan = bool(torch.isnan(flat).any().item())
        has_inf = bool(torch.isinf(flat).any().item())
        return (has_nan or has_inf), has_nan, has_inf
    except Exception:
        return False, False, False


def inspect_safetensors(path: Path) -> dict:
    """返回 safetensors 元信息；失败返回 {'error': ...}。"""
    if not HAS_SAFETENSORS:
        return {"error": "safetensors 未安装"}
    try:
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            return {"ok": True, "num_tensors": len(keys), "keys_sample": keys[:5]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


ROLE_REQUIRED = {
    "provenance": True, "ours_s1_recons": True, "ours_s1_captions": True, "ours_s1_ids": True,
    "gt_images": True, "gt_captions": True, "mindeye2_zavy": True, "sdxl_base": True,
    "sdxl_refiner": False,  # 可选对照
}

OURS_SUFFIXES = {
    "ours_s1_recons": "_all_enhancedrecons.pt",
    "ours_s1_captions": "_all_predcaptions.pt",
    "ours_s1_ids": "_all_ids.pt",
}


def check_entry(e: dict) -> dict:
    p = Path(e["local_path"])
    r = {
        "role": e["role"], "local_path": str(p), "remote_path": e["remote_path"],
        "repo_id": e["repo_id"],
        "required": e["required_or_optional"] != "optional",
        "exists": p.exists(), "size_ok": False, "sha_ok": False,
        "size_expected": 0, "size_actual": 0, "sha_expected": "", "sha_actual": "",
    }
    try:
        r["size_expected"] = int(e["expected_size"]) if str(e["expected_size"]).strip() else 0
    except ValueError:
        r["size_expected"] = 0
    r["sha_expected"] = str(e["expected_sha256"]).strip()
    if not p.exists():
        r["error"] = "file missing"
        return r
    r["size_actual"] = p.stat().st_size
    r["size_ok"] = (r["size_actual"] == r["size_expected"]) if r["size_expected"] else True
    if r["sha_expected"]:
        r["sha_actual"] = sha256_file(p)
        r["sha_ok"] = (r["sha_actual"] == r["sha_expected"])
    else:
        r["sha_ok"] = True
    return r


def by_role(results: list, role: str):
    return [r for r in results if r["role"] == role]


def deep_check_pt(role: str, path_str: str) -> dict:
    """对可信来源 .pt 文件做 CPU 加载与结构检查。"""
    out = {"role": role, "path": path_str, "ok": False}
    p = Path(path_str)
    if not p.exists():
        out["error"] = "missing"
        return out
    obj, err = load_pt(p, mmap=True)
    if err:
        # 回退非 mmap（小文件）
        obj, err2 = load_pt(p, mmap=False)
        if err2:
            out["error"] = f"{err}; fallback: {err2}"
            return out
    out["desc"] = describe_brief(obj)
    out["n_samples"] = n_samples_of(obj)
    out["ok"] = True
    return out


def check_ours_prefix(results) -> dict:
    names = {}
    for role, suffix in OURS_SUFFIXES.items():
        rr = by_role(results, role)
        if not rr:
            names[role] = None
            continue
        base = Path(rr[0]["local_path"]).name
        names[role] = base[: -len(suffix)] if base.endswith(suffix) else base
    prefixes = [v for v in names.values() if v]
    consistent = len(set(prefixes)) == 1
    return {"names": names, "consistent": consistent,
            "prefix": prefixes[0] if consistent and prefixes else None}


def check_alignment(deep, results) -> dict:
    a = {"ok": True, "issues": []}
    counts = {
        "ours_recons": deep.get("ours_s1_recons", {}).get("n_samples"),
        "ours_captions": deep.get("ours_s1_captions", {}).get("n_samples"),
        "ours_ids": deep.get("ours_s1_ids", {}).get("n_samples"),
        "gt_images": deep.get("gt_images", {}).get("n_samples"),
        "gt_captions": deep.get("gt_captions", {}).get("n_samples"),
    }
    a["counts"] = counts
    ours_counts = [counts["ours_recons"], counts["ours_captions"], counts["ours_ids"]]
    distinct = {x for x in ours_counts if x is not None}
    if len(distinct) > 1:
        a["ok"] = False
        a["issues"].append(f"Ours 三文件样本数不一致: {ours_counts}")
    if (counts["gt_images"] is not None and counts["gt_captions"] is not None
            and counts["gt_images"] != counts["gt_captions"]):
        a["ok"] = False
        a["issues"].append(f"GT images({counts['gt_images']}) 与 captions({counts['gt_captions']}) 数量不一致")

    rr = by_role(results, "ours_s1_ids")
    if rr and Path(rr[0]["local_path"]).exists():
        obj, err = load_pt(Path(rr[0]["local_path"]), mmap=False)
        if err:
            a["ok"] = False
            a["issues"].append(f"无法加载 Ours ID 文件: {err}")
        else:
            ids, err2 = extract_ids(obj)
            if err2:
                a["ok"] = False
                a["issues"].append(f"无法提取 Ours ID: {err2}")
            else:
                a["id_count"] = len(ids)
                a["id_unique"] = len(set(ids)) == len(ids)
                if not a["id_unique"]:
                    a["ok"] = False
                    a["issues"].append(f"Ours ID 存在 {len(ids) - len(set(ids))} 个重复")
                if counts["ours_ids"] is not None and len(ids) != counts["ours_ids"]:
                    a["ok"] = False
                    a["issues"].append(f"Ours ID 数量({len(ids)}) 与 n_samples({counts['ours_ids']}) 不一致")
                if counts["gt_images"]:
                    oob = [x for x in ids if x < 0 or x >= counts["gt_images"]]
                    if oob:
                        a["ok"] = False
                        a["issues"].append(f"Ours ID 超出 GT 范围 [0,{counts['gt_images']}): {len(oob)} 个")
                    else:
                        a["gt_aligned"] = True
                else:
                    a["ok"] = False
                    a["issues"].append("无法校验 ID 与 GT 对齐：缺少 GT images 样本数")
    else:
        a["ok"] = False
        a["issues"].append("缺少 Ours ID 文件，无法进行 ID 对齐")
    return a


def check_captions(results) -> dict:
    out = {"ok": True, "issues": []}
    for role in ("gt_captions", "ours_s1_captions"):
        rr = by_role(results, role)
        if not rr:
            continue
        p = Path(rr[0]["local_path"])
        if not p.exists():
            continue
        obj, err = load_pt(p, mmap=False)
        if err:
            out["ok"] = False
            out["issues"].append(f"{role} 无法加载: {err}")
            continue
        caps, err2 = extract_captions(obj)
        if err2:
            out["ok"] = False
            out["issues"].append(f"{role} 无法提取 caption: {err2}")
            continue
        empties = [i for i, c in enumerate(caps) if not str(c).strip()]
        out[f"{role}_total"] = len(caps)
        out[f"{role}_empty"] = len(empties)
        if empties:
            out["ok"] = False
            out["issues"].append(f"{role} 有 {len(empties)} 个空 caption")
    return out


def check_nonfinite_tensors(results) -> dict:
    out = {"ok": True, "issues": []}
    for role in ("ours_s1_recons", "gt_images"):
        rr = by_role(results, role)
        if not rr:
            continue
        p = Path(rr[0]["local_path"])
        if not p.exists():
            continue
        obj, err = load_pt(p, mmap=True)
        if err:
            obj, err = load_pt(p, mmap=False)
        if err or obj is None:
            out["ok"] = False
            out["issues"].append(f"{role} 无法加载以检查非有限值: {err}")
            continue
        t = obj if (torch is not None and torch.is_tensor(obj)) else None
        if t is None and isinstance(obj, dict):
            for v in obj.values():
                if torch is not None and torch.is_tensor(v):
                    t = v
                    break
        if t is None:
            out[f"{role}_checked"] = False
            out["issues"].append(f"{role} 非 tensor，跳过非有限值检查")
            continue
        nonfinite, has_nan, has_inf = check_nonfinite(t)
        out[f"{role}_nonfinite"] = nonfinite
        if nonfinite:
            out["ok"] = False
            out["issues"].append(f"{role} 存在非有限值 (nan={has_nan}, inf={has_inf})")
    return out


def check_provenance(results) -> dict:
    out = {"ok": True, "issues": [], "details": []}
    prov = by_role(results, "provenance")
    checksum_file = None
    for r in prov:
        p = Path(r["local_path"])
        out["details"].append(f"{p.name}: exists={p.exists()} size_ok={r['size_ok']} sha_ok={r['sha_ok']}")
        if not p.exists() or not r["size_ok"] or not r["sha_ok"]:
            out["ok"] = False
            out["issues"].append(f"provenance 文件不完整: {p.name}")
        if r["remote_path"].endswith("FINAL_OUTPUT_CHECKSUMS.sha256"):
            checksum_file = p
    out["checksum_contains_ours"] = {}
    if checksum_file and checksum_file.exists():
        content = checksum_file.read_text(encoding="utf-8", errors="replace")
        for role in ("ours_s1_recons", "ours_s1_captions", "ours_s1_ids"):
            rr = by_role(results, role)
            if not rr:
                continue
            sha = rr[0].get("sha_actual", "")
            in_file = (sha in content) if sha else None
            out["checksum_contains_ours"][role] = in_file
            if sha and not in_file:
                out["ok"] = False
                out["issues"].append(f"{role} 的 SHA256 未出现在 FINAL_OUTPUT_CHECKSUMS.sha256")
    elif checksum_file:
        out["ok"] = False
        out["issues"].append("FINAL_OUTPUT_CHECKSUMS.sha256 不存在")
    return out


def render_results_table(results) -> str:
    lines = ["| role | repo/path | exists | size | sha256 | required |",
             "|---|---|---|---|---|---|"]
    for r in results:
        size = f"{r['size_actual']}/{r['size_expected']}" + (" OK" if r["size_ok"] else " MISMATCH")
        sha = "OK" if r["sha_ok"] else ("MISMATCH" if r["sha_actual"] else "n/a")
        exists = "yes" if r["exists"] else "NO"
        lines.append(f"| {r['role']} | {r['repo_id']}/{r['remote_path']} | {exists} | {size} | {sha} | {r['required']} |")
    return "\n".join(lines)


def run_validation() -> int:
    entries = load_manifest(MANIFEST)
    if not entries:
        print(f"manifest 为空或不存在: {MANIFEST}")
        return 2
    results = [check_entry(e) for e in entries]

    deep = {}
    for role in ("ours_s1_recons", "ours_s1_captions", "ours_s1_ids", "gt_images", "gt_captions"):
        rr = by_role(results, role)
        if rr:
            deep[role] = deep_check_pt(role, rr[0]["local_path"])

    st = {}
    for role in ("mindeye2_zavy", "sdxl_base", "sdxl_refiner"):
        rr = by_role(results, role)
        if rr and Path(rr[0]["local_path"]).exists():
            st[role] = inspect_safetensors(Path(rr[0]["local_path"]))

    ours_prefix = check_ours_prefix(results)
    alignment = check_alignment(deep, results)
    captions = check_captions(results)
    nonfinite = check_nonfinite_tensors(results)
    provenance = check_provenance(results)

    missing_required = [r for r in results if r["required"] and not (r["exists"] and r["size_ok"] and r["sha_ok"])]
    optional_failed = [r for r in results if (not r["required"]) and not (r["exists"] and r["size_ok"] and r["sha_ok"])]
    core_ok = (not missing_required) and ours_prefix["consistent"] and alignment["ok"] and provenance["ok"]
    status = "READY_FOR_GPU" if core_ok else "BLOCKED_MISSING_REQUIRED_ASSETS"

    write_reports(results, deep, st, ours_prefix, alignment, captions, nonfinite,
                  provenance, missing_required, optional_failed, status)

    print(f"\n=== 验证完成 === 最终状态: {status}")
    print(f"必需文件缺失/损坏: {len(missing_required)}  可选文件失败: {len(optional_failed)}")
    for i in alignment.get("issues", []):
        print("  -", i)
    return 0 if status == "READY_FOR_GPU" else 1


def write_reports(results, deep, st, ours_prefix, alignment, captions, nonfinite,
                  provenance, missing_required, optional_failed, status):
    ts = utcnow_iso()

    # ---- DOWNLOAD_FINAL_REPORT.md ----
    dl = ["# Download Final Report", "", f"生成时间: {ts}", "", f"最终状态: **{status}**", ""]
    if missing_required:
        dl.append("## 缺失/损坏的必需文件")
        for r in missing_required:
            dl.append(f"- {r['role']}: {r['repo_id']}/{r['remote_path']} -> {r['local_path']}")
    else:
        dl.append("所有必需文件均已下载并通过大小/SHA256 校验。")
    if optional_failed:
        dl.append("")
        dl.append("## 可选文件失败（不影响主实验 READY 状态，但需单独记录）")
        for r in optional_failed:
            dl.append(f"- {r['role']}: {r['repo_id']}/{r['remote_path']} -> {r['local_path']}")
    dl += ["", "## 文件清单", "", render_results_table(results), ""]
    write_report("DOWNLOAD_FINAL_REPORT.md", "\n".join(dl) + "\n")

    # ---- ASSET_INTEGRITY_REPORT.md ----
    integ = ["# Asset Integrity Report", "", f"生成时间: {ts}", "", "## 大小与 SHA256", "",
             render_results_table(results), "", "## Safetensors 权重完整性", ""]
    for role, v in st.items():
        if v.get("ok"):
            integ.append(f"- {role}: OK，张量数={v['num_tensors']}，样例 keys={v.get('keys_sample')}")
        else:
            integ.append(f"- {role}: ERROR {v.get('error')}")
    integ += ["", "## .pt 文件结构", ""]
    for role, d in deep.items():
        if d.get("ok"):
            integ.append(f"- {role}: {d.get('desc')} (n_samples={d.get('n_samples')})")
        else:
            integ.append(f"- {role}: ERROR {d.get('error')}")
    integ += ["", "## 非有限值检查", ""]
    for k, v in nonfinite.items():
        if k.endswith("_nonfinite"):
            integ.append(f"- {k}: {v}")
    for i in nonfinite.get("issues", []):
        integ.append(f"- issue: {i}")
    integ += ["", "## 空 Caption 检查", ""]
    for k in ("gt_captions_total", "gt_captions_empty", "ours_s1_captions_total", "ours_s1_captions_empty"):
        if k in captions:
            integ.append(f"- {k}: {captions[k]}")
    for i in captions.get("issues", []):
        integ.append(f"- issue: {i}")
    write_report("ASSET_INTEGRITY_REPORT.md", "\n".join(integ) + "\n")

    # ---- ASSET_ALIGNMENT_REPORT.md ----
    align = ["# Asset Alignment Report", "", f"生成时间: {ts}", "", "## 样本数量", ""]
    for k, v in alignment.get("counts", {}).items():
        align.append(f"- {k}: {v}")
    align += ["", "## Ours run 前缀一致性", "",
              f"- 一致性: {ours_prefix['consistent']}", f"- 前缀: {ours_prefix.get('prefix')}"]
    for k, v in ours_prefix.get("names", {}).items():
        align.append(f"- {k}: {v}")
    align += ["", "## ID 检查", "",
              f"- ID 数量: {alignment.get('id_count')}",
              f"- ID 唯一: {alignment.get('id_unique')}",
              f"- 与 GT 对齐: {alignment.get('gt_aligned', False)}", ""]
    if alignment.get("issues"):
        align.append("## 问题")
        for i in alignment["issues"]:
            align.append(f"- {i}")
    write_report("ASSET_ALIGNMENT_REPORT.md", "\n".join(align) + "\n")

    # ---- MODEL_PROVENANCE.md ----
    prov = ["# Model Provenance", "", f"生成时间: {ts}", "", "## 来源文件", ""]
    for d in provenance.get("details", []):
        prov.append(f"- {d}")
    prov += ["", "## Ours S1 正式输出", "",
             f"- 来自同一正式 run: {ours_prefix['consistent']}",
             f"- run 前缀: {ours_prefix.get('prefix')}"]
    prov += ["", "## Ours 文件 SHA256 与 FINAL_OUTPUT_CHECKSUMS 匹配", ""]
    for role, ok in provenance.get("checksum_contains_ours", {}).items():
        prov.append(f"- {role}: {'匹配' if ok else ('不匹配' if ok is False else '无 SHA 可查')}")
    prov += ["", "## 权重来源", ""]
    for r in results:
        if r["role"] in ("mindeye2_zavy", "sdxl_base", "sdxl_refiner"):
            prov.append(f"- {r['role']}: repo={r['repo_id']} revision={r['revision']} "
                        f"sha256={r.get('sha_expected') or r.get('sha_actual')}")
    prov += ["", "## MindEye2 源代码", "",
             "- GitHub: MedARC-AI/MindEyeV2", "- commit: 76a2609243ba1bacb450e47e3dc78de1f78a9c00",
             "- 本地: source/MindEyeV2（按 commit 固定 tarball 获取）"]
    if provenance.get("issues"):
        prov += ["", "## 问题"]
        for i in provenance["issues"]:
            prov.append(f"- {i}")
    write_report("MODEL_PROVENANCE.md", "\n".join(prov) + "\n")


def main():
    rc = run_validation()
    sys.exit(rc)


if __name__ == "__main__":
    main()






