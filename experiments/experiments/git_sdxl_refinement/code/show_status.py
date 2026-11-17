#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 logs/status.json 与 logs/download.jsonl，展示当前下载状态。"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"))
STATUS = ROOT / "logs" / "status.json"
JSONL = ROOT / "logs" / "download.jsonl"
MANIFEST = ROOT / "manifests" / "download_manifest.tsv"


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


def human_duration(sec) -> str:
    sec = int(sec or 0)
    if sec <= 0:
        return "--"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    data = read_json(STATUS)
    if data is None:
        print("status.json 不存在 — 下载尚未开始或已结束。")
        if MANIFEST.exists():
            print("\n--- manifest 概览 ---")
            done = skipped = pending = 0
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()[1:]:
                cols = line.split("\t")
                if len(cols) < 10:
                    continue
                st = cols[9]
                if st == "DONE":
                    done += 1
                elif st == "FAILED_SKIPPED":
                    skipped += 1
                else:
                    pending += 1
            print(f"DONE={done}  FAILED_SKIPPED={skipped}  PENDING={pending}")
        return

    o = data.get("overall", {})
    c = data.get("current_file", {})
    sp = data.get("speed", {})

    lines = []
    lines.append(f"状态(Status)         : {data.get('status')}")
    if data.get("blocked"):
        lines.append(f"阻塞原因(Blocked)    : {data.get('blocked_reason')}")
    lines.append(f"总体进度(Overall)    : {o.get('progress_pct', 0):.2f}%  "
                 f"({human_bytes(o.get('downloaded_bytes', 0))} / {human_bytes(o.get('total_bytes', 0))})")
    lines.append(f"文件(Completed/Total): {o.get('completed_files')} / {o.get('total_files')}"
                 f"  失败(Failed): {o.get('failed_files', 0)}")
    lines.append(f"当前文件(Current)    : {c.get('name') or '-'}")
    lines.append(f"当前进度(Current)    : {c.get('progress_pct', 0):.2f}%  "
                 f"({human_bytes(c.get('downloaded_bytes', 0))} / {human_bytes(c.get('total_bytes', 0))})")
    lines.append(f"当前速度(Speed)      : {human_bytes(sp.get('current_bps', 0))}/s")
    lines.append(f"平均速度(Avg)        : {human_bytes(sp.get('avg_bps', 0))}/s")
    lines.append(f"剩余时间(ETA)        : {human_duration(sp.get('eta_seconds', 0))}")
    lines.append(f"重试次数(Retry)      : {o.get('retry_count', 0)}")

    # 最近事件
    events = []
    if JSONL.exists():
        try:
            with open(JSONL, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass
    print("\n".join(lines))
    if events:
        print("\n--- 最近事件 (Recent events) ---")
        for ev in events[-8:]:
            desc = ev.get("event")
            repo = ev.get("repo_id", "")
            rp = ev.get("remote_path", "")
            extra = ""
            if "strategy" in ev:
                extra = f" strategy={ev['strategy']}"
            if "error_type" in ev:
                extra += f" error_type={ev['error_type']}"
            print(f"  {ev.get('ts', '')}  {desc}  {repo}/{rp}{extra}")


if __name__ == "__main__":
    main()
