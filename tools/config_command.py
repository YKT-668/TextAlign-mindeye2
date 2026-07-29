#!/usr/bin/env python3
"""Render or execute a portable ConceptAlign YAML command configuration."""

import argparse
import os
import shlex
import subprocess

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("command"), list):
        raise ValueError("config must contain a command list")
    if not all(isinstance(item, str) for item in config["command"]):
        raise ValueError("every command item must be a string")
    env = config.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("env must be a mapping")
    return config


def render(config: dict) -> str:
    env = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in sorted(config.get("env", {}).items())
    )
    command = shlex.join(config["command"])
    return f"{env} {command}".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow execution of a configuration not marked status: verified.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print(render(config))
    if args.execute and config.get("status") != "verified" and not args.allow_unverified:
        raise SystemExit(
            "refusing to execute an unverified configuration; review it and pass "
            "--allow-unverified only after author confirmation"
        )
    if args.execute:
        environment = os.environ.copy()
        environment.update({key: str(value) for key, value in config.get("env", {}).items()})
        subprocess.run(config["command"], env=environment, check=True)


if __name__ == "__main__":
    main()
