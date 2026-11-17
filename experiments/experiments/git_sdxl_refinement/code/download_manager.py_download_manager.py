#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConceptAlign — GIT Caption-guided SDXL Img2Img 补充实验：HF 下载管理器。

按 manifests/download_manifest.tsv 顺序、单文件串行下载：
  - hf_hub_download + hf_xet（本机内存不足，不启用 HF_XET_HIGH_PERFORMANCE，使用默认自适应并发）
  - 自定义 tqdm_class 接收 HF 下载进度回调，统一字节统计
  - Rich 双进度条：按字节加权的总体进度 + 当前文件进度
  - 实时速度 / 平均速度 / ETA / 已完成文件数 / 重试次数 / 状态
  - 每 5s 写 logs/status.json；文件状态变化追加 logs/download.jsonl；全量日志 logs/download.log
  - 每文件最多一次自动修复重试，二次失败标记 FAILED_SKIPPED 并继续
  - 连续 180s 无有效进度判定疑似卡死
  - 下载成功后同盘硬链接到 assets/，避免双倍占用
"""
import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub import constants as hf_constants
from huggingface_hub import errors as hf_errors

from tqdm import tqdm as _tqdm

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

ROOT = Path(os.environ.get("ROOT", "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"))
MANIFEST_PATH = ROOT / "manifests" / "download_manifest.tsv"
LOG_DIR = ROOT / "logs"
STATUS_PATH = LOG_DIR / "status.json"
JSONL_PATH = LOG_DIR / "download.jsonl"
LOG_PATH = LOG_DIR / "download.log"

STALL_TIMEOUT = 180
STATUS_INTERVAL = 5

MANIFEST_COLUMNS = [
    "repo_id", "repo_type", "revision", "remote_path",
    "expected_size", "expected_sha256", "role", "required_or_optional",
    "local_path", "status", "attempts",
]


def utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_bytes(n) -> str:
    n = float(n or 0)
    if n == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if units[i] == "B":
        return f"{int(n)} B"
    return f"{n:.2f} {units[i]}"


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


class BlockingError(Exception):
    """需要人工介入、停止整个队列的错误（认证 / 磁盘）。"""


class ManifestEntry:
    def __init__(self, **kw):
        for k in MANIFEST_COLUMNS:
            setattr(self, k, kw.get(k, ""))

    @property
    def size(self) -> int:
        try:
            return int(self.expected_size) if str(self.expected_size).strip() else 0
        except (TypeError, ValueError):
            return 0

    @property
    def required(self) -> bool:
        return str(self.required_or_optional).strip().lower() != "optional"


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
            row = {header[i]: (cols[i] if i < len(cols) else "") for i in range(len(header))}
            row.setdefault("status", "PENDING")
            row.setdefault("attempts", "0")
            entries.append(ManifestEntry(**row))
    return entries


def save_manifest(path: Path, entries: list) -> None:
    tmp = path.with_suffix(".tsv.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for e in entries:
            f.write("\t".join(str(getattr(e, c, "")) for c in MANIFEST_COLUMNS) + "\n")
    os.replace(tmp, path)


# 供 tqdm 回调访问的全局管理器单例（下载主循环开始前设置）
_MANAGER = None


class _HFProgressBar(_tqdm):
    """hf_hub_download 的 tqdm_class 回调：不直接写终端，交给 Rich 统一渲染。

    xet 下载会创建两个 bar：reconstruction（bar_format 含 total_fmt，反映文件写入字节）
    和 transfer（网络字节）。只把 reconstruction（或 HTTP 单 bar）计入文件进度，
    transfer bar 仅用于刷新活跃时间（卡死检测）。
    """

    def __init__(self, *args, **kwargs):
        bar_format = str(kwargs.get("bar_format", "") or "")
        self._is_transfer_bar = bool(bar_format) and ("total_fmt" not in bar_format)
        self._bytes = 0  # 手动累计字节（disable=True 时 tqdm 不递增 self.n）
        kwargs["disable"] = True  # Rich 负责显示，tqdm 仅负责计数
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("unit_divisor", 1024)
        kwargs.setdefault("miniters", 1)
        super().__init__(*args, **kwargs)
        if _MANAGER is not None:
            _MANAGER._on_bar_init(self)

    def update(self, n: float = 1):
        self._bytes += int(n)
        if _MANAGER is not None:
            _MANAGER._on_bar_update(self)
        return True

    def close(self):
        if _MANAGER is not None:
            _MANAGER._on_bar_update(self)
        try:
            super().close()
        except Exception:
            pass


class DownloadManager:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.entries = load_manifest(manifest_path)
        self.console = Console()
        self.lock = threading.Lock()

        self.total_bytes = sum(e.size for e in self.entries)
        self.done_bytes = 0
        self.completed_files = 0
        self.total_files = len(self.entries)
        self.retry_count = 0
        self.status = "INIT"
        self.blocked = False
        self.blocked_reason = ""

        self.current_entry = None
        self.current_file = ""
        self.current_bytes = 0
        self.current_total = 0
        self.transfer_bytes = 0

        self.start_ts = time.time()
        self.last_progress_ts = time.time()
        self._byte_history = []  # [(ts, cumulative_file_bytes)] 用于总体/文件进度
        self._transfer_history = []  # [(ts, network_bytes)] 用于网络速度
        self.stall_logged = False

        self._live = None
        self._stop = threading.Event()
        self.logger = self._setup_logging()

        # 断点续传：本地已存在且大小一致的文件视为已完成
        for e in self.entries:
            if self._is_present(e):
                e.status = "DONE"
                self.completed_files += 1
                self.done_bytes += e.size
            elif e.status == "DONE":
                e.status = "PENDING"  # 本地缺失，需要重新下载

    # ------------------------------------------------------------------ setup
    def _setup_logging(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("conceptalign_download")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------------ helpers
    def _is_present(self, e) -> bool:
        p = Path(e.local_path)
        try:
            if not p.exists():
                return False
            if e.size > 0:
                return p.stat().st_size == e.size
            return p.stat().st_size > 0
        except OSError:
            return False

    def _err_str(self, ex) -> str:
        try:
            return f"{type(ex).__name__}: {ex}"
        except Exception:
            return type(ex).__name__

    def _jsonl(self, obj: dict) -> None:
        obj.setdefault("ts", utcnow_iso())
        try:
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _log(self, event: str, e=None, **kw) -> None:
        msg = f"event={event}"
        if e is not None:
            msg += f" repo={e.repo_id} path={e.remote_path} local={e.local_path}"
        for k, v in kw.items():
            msg += f" {k}={v}"
        try:
            self.logger.info(msg)
        except Exception:
            pass


    # ------------------------------------------------------------------ progress callback
    def _on_bar_init(self, bar):
        with self.lock:
            if not getattr(bar, "_is_transfer_bar", False):
                self.current_total = int(getattr(bar, "total", 0) or self.current_total or 0)
                self.current_bytes = int(getattr(bar, "_bytes", 0) or 0)
            self.last_progress_ts = time.time()
            self.stall_logged = False

    def _on_bar_update(self, bar):
        with self.lock:
            self.last_progress_ts = time.time()
            self.stall_logged = False
            if getattr(bar, "_is_transfer_bar", False):
                # 网络字节 bar：刷新活跃时间 + 记录网络速度
                self.transfer_bytes = int(getattr(bar, "_bytes", 0) or 0)
                self._record_transfer_history()
                return
            self.current_bytes = int(getattr(bar, "_bytes", 0) or 0)
            self.current_total = int(getattr(bar, "total", 0) or self.current_total or 0)
            self._record_history()

    def _record_history(self):
        now = time.time()
        cum = self.done_bytes + self.current_bytes
        if self._byte_history and (now - self._byte_history[-1][0]) < 1.0:
            return
        self._byte_history.append((now, cum))
        cutoff = now - 120
        self._byte_history = [(t, b) for t, b in self._byte_history if t >= cutoff]

    def _record_transfer_history(self):
        now = time.time()
        if self._transfer_history and (now - self._transfer_history[-1][0]) < 1.0:
            return
        self._transfer_history.append((now, self.transfer_bytes))
        cutoff = now - 120
        self._transfer_history = [(t, b) for t, b in self._transfer_history if t >= cutoff]

    # ------------------------------------------------------------------ stats
    def _stats(self) -> dict:
        with self.lock:
            downloaded = self.done_bytes + self.current_bytes
            total = self.total_bytes
            pct = (downloaded / total * 100.0) if total else 0.0
            elapsed = max(time.time() - self.start_ts, 1e-6)
            avg_bps = downloaded / elapsed
            cur_bps = 0.0
            # 当前速度优先用网络字节（transfer 更新频繁），否则退回文件字节
            hist = self._transfer_history or self._byte_history
            if len(hist) >= 2:
                dt = hist[-1][0] - hist[0][0]
                if dt > 0:
                    cur_bps = (hist[-1][1] - hist[0][1]) / dt
            speed = cur_bps if cur_bps > 0 else avg_bps
            remaining = max(total - downloaded, 0)
            eta = (remaining / speed) if speed > 0 else 0.0
            return {
                "downloaded": downloaded, "total": total, "pct": pct,
                "completed_files": self.completed_files, "total_files": self.total_files,
                "retry_count": self.retry_count, "status": self.status,
                "current_file": self.current_file,
                "current_bytes": self.current_bytes, "current_total": self.current_total,
                "transfer_bytes": self.transfer_bytes,
                "avg_bps": avg_bps, "cur_bps": cur_bps, "eta": eta,
            }

    # ------------------------------------------------------------------ render
    def _render(self):
        s = self._stats()
        overall = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.percentage:>5.1f}%"),
            console=self.console,
        )
        overall.add_task("Overall", total=max(s["total"], 1), completed=s["downloaded"])

        cur_pct = (s["current_bytes"] / s["current_total"] * 100.0) if s["current_total"] else 0.0
        curfile = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.percentage:>5.1f}%"),
            console=self.console,
        )
        curfile.add_task("Current", total=max(s["current_total"], 1), completed=s["current_bytes"])

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", justify="right")
        table.add_column(style="")
        table.add_row("Current File", s["current_file"] or "-")
        table.add_row("Current File Progress",
                      f"{human_bytes(s['current_bytes'])} / {human_bytes(s['current_total'])} ({cur_pct:.1f}%)")
        table.add_row("Network Bytes (xet)", f"{human_bytes(s['transfer_bytes'])}")
        table.add_row("Downloaded / Total", f"{human_bytes(s['downloaded'])} / {human_bytes(s['total'])}")
        table.add_row("Current Speed", f"{human_bytes(s['cur_bps'])}/s")
        table.add_row("Average Speed", f"{human_bytes(s['avg_bps'])}/s")
        table.add_row("Estimated Remaining Time", human_duration(s["eta"]))
        table.add_row("Completed / Total Files", f"{s['completed_files']} / {s['total_files']}")
        table.add_row("Retry Count", str(s["retry_count"]))
        table.add_row("Current Status", s["status"])

        return Group(
            Panel(overall, title="Overall Progress", border_style="cyan"),
            Panel(curfile, title="Current File Progress", border_style="magenta"),
            Panel(table, title="Statistics", border_style="green"),
        )


    # ------------------------------------------------------------------ download
    def download_one(self, e):
        if e.status == "DONE":
            return
        if self._is_present(e):
            self._mark_done(e, "present-already")
            return
        self.status = "DOWNLOADING"
        self.current_entry = e
        self.current_file = e.local_path
        self.current_bytes = 0
        self.current_total = e.size
        self.transfer_bytes = 0
        self._byte_history = []
        self._transfer_history = []
        self.last_progress_ts = time.time()
        self.stall_logged = False
        e.attempts = int(e.attempts or 0)
        e.status = "DOWNLOADING"
        self._jsonl({"event": "file_started", "repo_id": e.repo_id,
                     "remote_path": e.remote_path, "local_path": e.local_path, "size": e.size})
        self._log("file_started", e)

        last_err = None
        for attempt in (1, 2):
            e.attempts = attempt
            try:
                path = self._hf_download(e)
                self._hardlink(path, e)
                self._mark_done(e, "downloaded")
                return
            except Exception as ex:
                last_err = ex
                kind = self._classify(ex)
                self.retry_count += 1
                self._jsonl({"event": "file_failed", "repo_id": e.repo_id,
                             "remote_path": e.remote_path, "attempt": attempt,
                             "error_type": kind, "error": self._err_str(ex)})
                self._log("file_failed", e, attempt=attempt, kind=kind, err=self._err_str(ex))
                if kind in ("auth_401", "auth_403", "auth_token_missing", "disk_full"):
                    raise BlockingError(f"{kind} on {e.repo_id}/{e.remote_path}: {self._err_str(ex)}")
                if attempt == 1:
                    strategy = self._apply_strategy(kind, ex)
                    self._jsonl({"event": "file_retry", "repo_id": e.repo_id,
                                 "remote_path": e.remote_path, "strategy": strategy,
                                 "error_type": kind})
                    self._log("file_retry", e, strategy=strategy, kind=kind)
                else:
                    self._mark_failed_skipped(e, last_err)
                    self._cleanup_temp(e)
                    return
        self._mark_failed_skipped(e, last_err)
        self._cleanup_temp(e)

    def _hf_download(self, e) -> str:
        return hf_hub_download(
            repo_id=e.repo_id,
            filename=e.remote_path,
            repo_type=(e.repo_type or None),
            revision=(e.revision or None),
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            token=True,
            tqdm_class=_HFProgressBar,
            etag_timeout=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "60") or 60),
        )

    def _hardlink(self, cached: str, e):
        real = os.path.realpath(cached)
        dst = Path(e.local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except OSError:
                pass
        try:
            os.link(real, str(dst))
        except OSError:
            shutil.copyfile(real, str(dst))
        try:
            actual = dst.stat().st_size
        except OSError:
            actual = -1
        if e.size > 0 and actual != e.size:
            self._log("size_mismatch_after_hardlink", e, actual=actual)


    def _classify(self, ex) -> str:
        if isinstance(ex, hf_errors.XetDownloadError):
            return "xet_cas"
        if isinstance(ex, hf_errors.LocalTokenNotFoundError):
            return "auth_token_missing"
        if isinstance(ex, hf_errors.HfHubHTTPError):
            code = getattr(getattr(ex, "response", None), "status_code", None)
            if code == 401:
                return "auth_401"
            if code == 403:
                return "auth_403"
            if code == 429:
                return "rate_429"
            return f"http_{code}" if code else "http"
        if isinstance(ex, TimeoutError):
            return "timeout"
        if "timeout" in type(ex).__name__.lower() or "timed" in type(ex).__name__.lower():
            return "timeout"
        if isinstance(ex, OSError):
            if getattr(ex, "errno", None) == 28:
                return "disk_full"
            return "oserror"
        return "unknown"

    def _apply_strategy(self, kind: str, ex) -> str:
        if kind == "timeout":
            try:
                hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = max(int(hf_constants.HF_HUB_DOWNLOAD_TIMEOUT or 10), 600)
            except Exception:
                pass
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
            return "increase-timeout-to-600s"
        if kind == "xet_cas":
            hf_constants.HF_HUB_DISABLE_XET = True
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            return "disable-xet-use-http"
        if kind == "rate_429":
            self.status = "WAITING_429"
            time.sleep(60)
            self.status = "DOWNLOADING"
            return "wait-60s"
        time.sleep(5)
        return "generic-retry-after-5s"

    def _mark_done(self, e, how: str):
        with self.lock:
            e.status = "DONE"
            self.completed_files += 1
            self.done_bytes += e.size
            self.current_bytes = 0
            self.current_total = 0
        self.status = "RUNNING"
        self._jsonl({"event": "file_done", "repo_id": e.repo_id,
                     "remote_path": e.remote_path, "local_path": e.local_path,
                     "size": e.size, "how": how})
        self._log("file_done", e, how=how)

    def _mark_failed_skipped(self, e, ex):
        with self.lock:
            e.status = "FAILED_SKIPPED"
            self.current_bytes = 0
        self._jsonl({"event": "file_failed_skipped", "repo_id": e.repo_id,
                     "remote_path": e.remote_path, "local_path": e.local_path,
                     "error": self._err_str(ex)})
        self._log("file_failed_skipped", e, err=self._err_str(ex))

    def _cleanup_temp(self, e):
        tmp = os.environ.get("TMPDIR") or "/tmp"
        patterns = [e.remote_path.replace("/", "_"), e.remote_path.split("/")[-1],
                    e.repo_id.replace("/", "__")]
        removed = 0
        if os.path.isdir(tmp):
            for fn in os.listdir(tmp):
                if any(p in fn for p in patterns if p):
                    try:
                        os.remove(os.path.join(tmp, fn))
                        removed += 1
                    except OSError:
                        pass
        self._log("cleanup_temp", e, removed=removed, tmp=tmp)


    # ------------------------------------------------------------------ status / logs
    def _status_writer(self):
        while not self._stop.is_set():
            self._write_status_json()
            self._check_stall()
            self._stop.wait(STATUS_INTERVAL)

    def _write_status_json(self):
        s = self._stats()
        payload = {
            "ts": utcnow_iso(),
            "root": str(ROOT),
            "status": self.status,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "overall": {
                "total_bytes": s["total"],
                "downloaded_bytes": s["downloaded"],
                "progress_pct": round(s["pct"], 3),
                "total_files": s["total_files"],
                "completed_files": s["completed_files"],
                "failed_files": sum(1 for e in self.entries if e.status == "FAILED_SKIPPED"),
                "retry_count": s["retry_count"],
            },
            "current_file": {
                "name": s["current_file"],
                "downloaded_bytes": s["current_bytes"],
                "total_bytes": s["current_total"],
                "transfer_bytes": s["transfer_bytes"],
                "progress_pct": round(
                    (s["current_bytes"] / s["current_total"] * 100.0) if s["current_total"] else 0.0, 3),
            },
            "speed": {
                "current_bps": round(s["cur_bps"], 2),
                "avg_bps": round(s["avg_bps"], 2),
                "eta_seconds": round(s["eta"], 1),
            },
        }
        tmp = STATUS_PATH.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATUS_PATH)
        except OSError:
            pass

    def _check_stall(self):
        with self.lock:
            if (self.status == "DOWNLOADING" and self.current_total > 0
                    and self.current_bytes < self.current_total
                    and (time.time() - self.last_progress_ts) > STALL_TIMEOUT
                    and not self.stall_logged):
                self.stall_logged = True
                e = self.current_entry
                info = {"repo_id": "", "remote_path": ""}
                if e is not None:
                    info = {"repo_id": e.repo_id, "remote_path": e.remote_path}
                self._jsonl({"event": "suspected_stall", **info,
                             "idle_seconds": round(time.time() - self.last_progress_ts, 1)})
                self._log("suspected_stall", e, idle_seconds=round(time.time() - self.last_progress_ts, 1))

    # ------------------------------------------------------------------ run
    def run(self):
        global _MANAGER
        _MANAGER = self
        self._stop = threading.Event()
        self.status = "RUNNING"
        self._jsonl({"event": "run_started", "total_files": self.total_files,
                     "total_bytes": self.total_bytes})
        self._log("run_started", total_files=self.total_files, total_bytes=self.total_bytes)

        writer = threading.Thread(target=self._status_writer, daemon=True)
        writer.start()

        try:
            with Live(self._render(), console=self.console, refresh_per_second=4, screen=False) as live:
                self._live = live
                for e in self.entries:
                    if e.status == "DONE":
                        continue
                    self.current_entry = e
                    try:
                        self.download_one(e)
                    except BlockingError as ex:
                        self.blocked = True
                        self.blocked_reason = str(ex)
                        self.status = "BLOCKED"
                        self._jsonl({"event": "blocked", "reason": str(ex)})
                        self._log("blocked", reason=str(ex))
                        live.update(self._render())
                        break
                    finally:
                        self.current_entry = None
                    live.update(self._render())
                if not self.blocked:
                    self.status = "FINISHED"
                live.update(self._render())
        finally:
            self._stop.set()
            self._write_status_json()
            save_manifest(self.manifest_path, self.entries)
            self._jsonl({"event": "run_finished", "status": self.status,
                         "blocked_reason": self.blocked_reason})
            self._log("run_finished", status=self.status)
            self._print_summary()

        # 下载结束后自动执行第九节的数据检查
        self._run_validation()

    def _print_summary(self):
        s = self._stats()
        failed = [e for e in self.entries if e.status == "FAILED_SKIPPED"]
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold]Download run finished[/bold]\n"
            f"Status: [bold]{self.status}[/bold]\n"
            f"Files: {s['completed_files']}/{s['total_files']} completed\n"
            f"Bytes: {human_bytes(s['downloaded'])} / {human_bytes(s['total'])}\n"
            f"Failed/Skipped: {len(failed)}",
            title="Summary", border_style="cyan"))
        for e in failed:
            self.console.print(f"  [red]FAILED[/red] {e.repo_id}/{e.remote_path} -> {e.local_path}")
        if self.blocked:
            self.console.print(f"  [red]BLOCKED[/red]: {self.blocked_reason}")

    def _run_validation(self):
        """下载结束后自动执行数据完整性 + ID 对齐检查（第九节）。"""
        import subprocess
        script = ROOT / "scripts" / "validate_downloads.py"
        if not script.exists():
            return
        self._log("validation_started", script=str(script))
        self.status = "VALIDATING"
        self._write_status_json()
        try:
            rc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=10800,
            )
            self._log("validation_finished", rc=rc.returncode)
            for line in (rc.stdout or "").strip().splitlines()[-30:]:
                self._log("validation", line=line)
            for line in (rc.stderr or "").strip().splitlines()[-10:]:
                self._log("validation_stderr", line=line)
        except Exception as e:
            self._log("validation_error", err=self._err_str(e))
        finally:
            self.status = "VALIDATED"
            self._write_status_json()


def main():
    ap = argparse.ArgumentParser(description="ConceptAlign HF download manager")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    os.environ["ROOT"] = args.root
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mgr = DownloadManager(Path(args.manifest))
    mgr.run()


if __name__ == "__main__":
    main()

