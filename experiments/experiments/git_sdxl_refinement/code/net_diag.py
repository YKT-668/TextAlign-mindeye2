#!/usr/bin/env python3
"""网络线路诊断：DNS / HTTPS / 响应时间 / 重定向，输出到 reports/_net_test.txt。"""
import socket
import ssl
import time
import urllib.request
from pathlib import Path

OUT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/reports/_net_test.txt")

HOSTS = [
    "huggingface.co",
    "cas-server.xethub.hf.co",
    "transfer.xethub.hf.co",
    "cdn-lfs-us-1.hf.co",
    "hf-mirror.com",
]

lines = []


def dns(host):
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos})
        return f"OK {ips[:4]}"
    except Exception as e:
        return f"FAIL {type(e).__name__} {e}"


def https(host):
    """HTTPS GET 根路径，返回 (状态, 耗时, 重定向目标, 下载速度采样)。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    t0 = time.time()
    try:
        req = urllib.request.Request(f"https://{host}/", headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        code = resp.status
        final = resp.geturl()
        dt = time.time() - t0
        # 采样前 1MB 的下载速度（对首页可能很小）
        data = resp.read(1024 * 1024)
        dt2 = time.time() - t0
        spd = len(data) / dt2 if dt2 > 0 else 0
        return f"code={code} t={dt:.2f}s final={final} sample={len(data)}B {spd/1024:.0f}KB/s"
    except Exception as e:
        dt = time.time() - t0
        return f"FAIL {type(e).__name__} t={dt:.2f}s {str(e)[:120]}"


for h in HOSTS:
    lines.append(f"== {h} ==")
    lines.append(f"  DNS: {dns(h)}")
    lines.append(f"  HTTPS: {https(h)}")

# 额外：测试 hf-mirror 的 HF 镜像端点响应
lines.append("== hf-mirror.com HF API 端点 ==")
for path in ["/api/models/pscotti/mindeyev2", "/pscotti/mindeyev2/resolve/main/README.md"]:
    try:
        t0 = time.time()
        req = urllib.request.Request(f"https://hf-mirror.com{path}", headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15, context=ssl._create_unverified_context())
        code = resp.status
        dt = time.time() - t0
        lines.append(f"  {path}: code={code} t={dt:.2f}s")
    except Exception as e:
        lines.append(f"  {path}: FAIL {type(e).__name__} {str(e)[:100]}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size)
