"""Regenerate requirements.lock for the container target (CPython 3.11, Linux x86_64).

`pip freeze` on a developer machine can pin wheels that do not exist for the deployment target
(e.g. a package that needs Python 3.12). This resolves requirements-dev.txt for the target with
pip's resolver and writes every resolved distribution with its exact version.

Usage: python scripts/lock.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ["--python-version", "3.11", "--implementation", "cp", "--abi", "cp311", "--abi", "abi3", "--abi", "none"]
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64", "linux_x86_64", "any"]
HEADER = [
    "# Exact versions for CPython 3.11 on Linux x86_64 (the Docker image and CI).",
    "# Regenerate with: python scripts/lock.py",
    "--extra-index-url https://download.pytorch.org/whl/cpu",
]


def main() -> int:
    """Resolve and write the lock file."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--quiet",
            "--only-binary=:all:",
            "--target",
            str(Path(tmp) / "t"),
            "--report",
            str(report),
            *TARGET,
            *[a for p in PLATFORMS for a in ("--platform", p)],
            "-r",
            str(ROOT / "requirements-dev.txt"),
        ]
        subprocess.run(cmd, check=True)
        installs = json.loads(report.read_text(encoding="utf-8"))["install"]
    pins = sorted(f"{i['metadata']['name']}=={i['metadata']['version'].split('+')[0]}" for i in installs)
    (ROOT / "requirements.lock").write_text("\n".join(HEADER + pins) + "\n", encoding="utf-8")
    print(f"requirements.lock: {len(pins)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
