"""Bound the existing pinned model downloader and retain its exit status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.output / "source-manifest.json").read_text())
    command = [sys.executable, str(args.assets / "download_model.py"), "--repo", receipt["repo"],
               "--output", str(args.output)]
    try:
        result = subprocess.run(command, timeout=3600)
        status = {"exit_code": result.returncode, "status": "verified" if result.returncode == 0 else "failed"}
    except subprocess.TimeoutExpired:
        status = {"exit_code": 124, "status": "timed_out"}
    status["finished_unix"] = time.time()
    (args.assets / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
