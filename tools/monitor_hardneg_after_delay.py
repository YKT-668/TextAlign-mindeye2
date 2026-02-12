#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
	try:
		out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
		return 0, out
	except subprocess.CalledProcessError as e:
		return int(e.returncode), (e.output or "")
	except FileNotFoundError:
		return 127, "command not found: " + (cmd[0] if cmd else "")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--delay_sec", type=int, default=600)
	ap.add_argument(
		"--process_grep",
		type=str,
		default=r"gen_shared982_hardneg_from_evals\\.py",
		help="Regex passed to pgrep -af",
	)
	ap.add_argument(
		"--heartbeat",
		type=str,
		default="cache/hardneg/shared982_hardneg_heartbeat.json",
	)
	ap.add_argument(
		"--run_log",
		type=str,
		default="cache/hardneg/shared982_hardneg_run.log",
	)
	ap.add_argument(
		"--out",
		type=str,
		default="cache/hardneg/shared982_hardneg_monitor_10min.log",
	)
	args = ap.parse_args()

	time.sleep(max(0, int(args.delay_sec)))

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	lines: list[str] = []
	lines.append("== hardneg 10min check ==")
	lines.append(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')}")

	# Process check
	rc, proc_out = _run(["pgrep", "-af", args.process_grep])
	lines.append("-- process --")
	lines.append(f"pgrep_rc={rc}")
	lines.append(proc_out.strip() or "(no match)")

	# Heartbeat check
	hb_path = Path(args.heartbeat)
	lines.append("-- heartbeat --")
	if hb_path.is_file():
		try:
			obj = json.loads(hb_path.read_text(encoding="utf-8"))
			ts = float(obj.get("ts", 0.0))
			age = time.time() - ts
			lines.append(f"heartbeat_age_sec={age:.2f}")
			lines.append("heartbeat_json=" + json.dumps(obj, ensure_ascii=False))
		except Exception as e:
			lines.append(f"heartbeat_parse_error={type(e).__name__}: {e}")
	else:
		lines.append("heartbeat_missing")

	# GPU snapshot
	lines.append("-- nvidia-smi compute --")
	rc, smi_out = _run(
		[
			"nvidia-smi",
			"--query-compute-apps=pid,process_name,used_memory",
			"--format=csv,noheader,nounits",
		]
	)
	lines.append(f"nvidia_smi_rc={rc}")
	lines.append(smi_out.strip() or "(empty)")

	# Log tail
	lines.append("-- run_log tail --")
	log_path = Path(args.run_log)
	if log_path.is_file():
		rc, tail_out = _run(["tail", "-n", "80", str(log_path)])
		lines.append(f"tail_rc={rc}")
		lines.append(tail_out.strip() or "(empty)")
	else:
		lines.append("run_log_missing")

	lines.append("== end ==\n")
	out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
