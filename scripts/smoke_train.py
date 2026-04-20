from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-m",
        "pacman_rl.train",
        "--layout-dir",
        "layouts",
        "--device",
        "cpu",
        "--batch-size",
        "32",
        "--updates",
        "2",
        "--run-dir",
        "runs/smoke",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
