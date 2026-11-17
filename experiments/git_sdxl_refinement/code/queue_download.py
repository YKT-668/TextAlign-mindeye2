#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多 Worker 队列下载器（线程）。断点续传、SHA 校验、硬链接、worker 状态日志。"""
import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get("ROOT", "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"))
LOG_DIR = ROOT / "logs"
STATUS = LOG_DIR / "status_queue.json"
WORKER_STATUS = LOG_DIR / "worker_status.json"
QUEUE_LOG = LOG_DIR / "download_queue.log"

_lock = threading.Lock()
_entries = []
_done_bytes = 0
_total_bytes = 0
_worker_state = {}


def human(n):
    n = float(n or 0)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{int(n)}B" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(QUEUE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def is_present(e):
    p = Path(e["local_path"])
    try:
        if not p.exists():
            return False
        sz = int(e["expected_size"])
        return p.stat().st_size == sz if sz else p.stat().st_size > 0
    except Exception:
        return False


def claim_next():
    with _lock:
        for e in _entries:
            if e["status"] == "PENDING" and not is_present(e):
                e["status"] = "DOWNLOADING"
                return e
    return None


def download_one(e, wid):
    global _done_bytes
    p = Path(e["local_path"])
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=e["repo_id"], filename=e["remote_path"],
            repo_type=(e["repo_type"] or None), revision=(e["revision"] or None),
            cache_dir=os.environ.get("HF_HUB_CACHE"), token=True,
        )
        real = os.path.realpath(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() or p.is_symlink():
            p.unlink()
        try:
            os.link(real, str(p))
        except OSError:
            shutil.copyfile(real, str(p))
        if e["expected_sha256"]:
            act = sha256_file(p)
            if act.lower() != e["expected_sha256"].lower():
                log(f"[W{wid}] SHA MISMATCH {e['remote_path']}")
                with _lock:
                    e["status"] = "FAILED"
                return
        with _lock:
            e["status"] = "DONE"
            _done_bytes += int(e["expected_size"])
        log(f"[W{wid}] DONE {e['repo_id']}/{e['remote_path']} {e['expected_size']}B in {time.time()-t0:.0f}s")
    except Exception as ex:
        with _lock:
            e["attempts"] = str(int(e["attempts"] or 0) + 1)
            e["status"] = "FAILED" if int(e["attempts"]) >= 3 else "PENDING"
        log(f"[W{wid}] ERR {e['remote_path']} {type(ex).__name__}: {str(ex)[:150]}")


def worker(wid):
    while True:
        e = claim_next()
        if e is None:
            break
        _worker_state[wid] = {"file": e["remote_path"]}
        download_one(e, wid)
        _worker_state[wid] = {"file": None}


def status_thread():
    while True:
        with _lock:
            done = sum(1 for e in _entries if e["status"] == "DONE")
            failed = sum(1 for e in _entries if e["status"] == "FAILED")
            dl = sum(1 for e in _entries if e["status"] == "DOWNLOADING")
            pend = sum(1 for e in _entries if e["status"] == "PENDING")
            payload = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "DOWNLOADING" if dl else ("FINISHED" if not pend else "PENDING"),
                "total_bytes": _total_bytes, "done_bytes": _done_bytes,
                "files": {"done": done, "failed": failed, "downloading": dl, "pending": pend},
            }
        tmp = STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(tmp, STATUS)
        WORKER_STATUS.write_text(json.dumps(_worker_state, ensure_ascii=False, indent=2))
        time.sleep(5)


def load_manifest(path):
    global _entries, _total_bytes, _done_bytes
    with open(path, encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.strip() for c in cols]
                continue
            e = {header[i]: (cols[i] if i < len(cols) else "") for i in range(len(header))}
            e.setdefault("status", "PENDING")
            e.setdefault("attempts", "0")
            _entries.append(e)
            try:
                _total_bytes += int(e["expected_size"])
            except Exception:
                pass
    for e in _entries:
        if is_present(e) and e["status"] != "DONE":
            e["status"] = "DONE"
            _done_bytes += int(e["expected_size"] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    load_manifest(args.manifest)
    log(f"start workers={args.workers} files={len(_entries)} total={human(_total_bytes)}")
    threading.Thread(target=status_thread, daemon=True).start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with _lock:
        done = sum(1 for e in _entries if e["status"] == "DONE")
        failed = sum(1 for e in _entries if e["status"] == "FAILED")
    log(f"finished done={done} failed={failed}")
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FINISHED", "total_bytes": _total_bytes, "done_bytes": _done_bytes,
        "files": {"done": done, "failed": failed},
    }
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

