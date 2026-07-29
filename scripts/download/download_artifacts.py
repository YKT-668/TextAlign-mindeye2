#!/usr/bin/env python3
"""Print or run pinned Hugging Face download commands."""

import argparse
import json
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--mapping", default="configs/artifacts.json")
    parser.add_argument("--local-dir", default="artifacts")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with open(args.mapping, encoding="utf-8") as handle:
        mapping = json.load(handle)
    item = mapping["artifacts"][args.artifact]
    command = [
        "hf",
        "download",
        item["repo_id"],
        item["path"],
        f"--repo-type={item['repo_type']}",
        f"--revision={item['revision']}",
        f"--local-dir={args.local_dir}",
    ]
    print(shlex.join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
