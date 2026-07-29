#!/usr/bin/env python3
"""Calculate or verify SHA-256 checksums without external dependencies."""

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect")
    args = parser.parse_args()
    actual = digest(args.path)
    print(actual)
    if args.expect and actual.lower() != args.expect.lower():
        raise SystemExit("checksum mismatch")


if __name__ == "__main__":
    main()
