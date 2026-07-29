#!/usr/bin/env python3
"""Run the dependency-light public release smoke suite."""

import subprocess
import sys


raise SystemExit(
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/smoke", "-v"]
    ).returncode
)
