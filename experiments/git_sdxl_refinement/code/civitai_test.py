#!/usr/bin/env python3
"""测试 Civitai 下载链路：重定向、最终域名、10MB 范围下载速度。输出到 reports/_civitai_test.txt。"""
import ssl
import time
import urllib.request
from pathlib import Path

OUT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/reports/_civitai_test.txt")
URL = "https://civitai.com/api/download/models/252831?fileId=195951"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

L = []
t0 = time.time()
try:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-10485759"})
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    final = resp.geturl()
    code = resp.status
    t_connect = time.time() - t0
    data = resp.read(1024 * 1024)
    dt = time.time() - t0
    L.append(f"code={code} t_first={t_connect:.2f}s final={final}")
    L.append(f"downloaded={len(data)}B in {dt:.2f}s -> {len(data)/dt/1024:.1f} KiB/s")
except Exception as e:
    L.append(f"FAIL {type(e).__name__} {str(e)[:200]}")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
