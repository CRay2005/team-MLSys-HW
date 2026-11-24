"""Upload a verified HW3 harness and start a bounded, detached server smoke suite."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "homework-3/scripts/run_offloading.py", "homework-3/scripts/compare_results.py",
    "homework-3/scripts/server_smoke.py", "homework-3/tests/test_offloading.py",
    "homework-3/tests/test_host_budget_contract.py", "homework-2/scripts/run_model.py",
    "homework-2/scripts/npu_occupancy.py", "homework-2/cases.json",
    "homework-2/metadata/Qwen3.5-0.8B/config.json",
    "homework-3/runtime/sitecustomize.py", "homework-3/runtime/runtime_observer.py",
    "homework-3/scripts/summarize_trace.py", "homework-3/tests/test_observer.py",
    "homework-2/metadata/Qwen3.5-4B/config.json",
]


def build_bundle(destination):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hw2_bundle", ROOT / FILES[5])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    trees = helper.project_source_fingerprints()
    if any("error" in tree for tree in trees.values()):
        raise RuntimeError("Local runtime source missing")
    manifest = {"files": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES},
                "expected_runtime_trees": trees}
    manifest_path = destination / "deployment-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination / "harness.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in FILES:
            tar.add(ROOT / name, arcname=name, recursive=False)
        tar.add(manifest_path, arcname=manifest_path.name, recursive=False)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Existing course SSH user@host")
    parser.add_argument("--device", type=int, choices=range(8), default=1,
                        help="Previously verified course physical chip ID; refuses occupied chip")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--suite", choices=("smoke", "trace", "matrix", "stress"), default="smoke")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-size", choices=("0.8B", "4B"), default="0.8B")
    parser.add_argument("--after-run", help="Queue behind this successful run; waits at most 30 minutes")
    args = parser.parse_args()
    if args.host.startswith("-") or any(c.isspace() for c in args.host):
        parser.error("Invalid SSH destination")
    if args.tp == 2 and args.device % 2:
        parser.error("TP2 requires an even first chip: --device 0, 2, 4, or 6")
    if args.after_run and not re.fullmatch(r"hw3-smoke-\d{8}T\d{6}Z-[a-f0-9]{8}", args.after_run):
        parser.error("Invalid predecessor run label")
    label = datetime.now(timezone.utc).strftime("hw3-smoke-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    local = ROOT / ".local-review" / label
    local.mkdir(parents=True, exist_ok=False)
    archive = build_bundle(local)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"Local package: {local}", flush=True)
    if args.prepare_only:
        print("Prepared only; no server connection or experiment started.")
        return
    common = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
              "-o", "IdentitiesOnly=yes", "-i", str(Path.home() / ".ssh/id_ed25519"),
              "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10",
              "-o", "ServerAliveCountMax=2"]
    ssh = ["ssh", "-p", "2222", *common, args.host]
    # The name is generated locally; no shell interpolation of user-supplied paths.
    remote = f"MLSys-HW/{label}"
    subprocess.run([*ssh, shlex.join(["mkdir", "--", remote])], check=True, timeout=45)
    subprocess.run(["scp", "-P", "2222", *common, str(archive),
                    f"{args.host}:{remote}/harness.tar.gz"], check=True, timeout=120)
    bootstrap = '''import hashlib, json, os, pathlib, subprocess, tarfile
root = pathlib.Path.home() / "MLSys-HW" / LABEL
archive = root / "harness.tar.gz"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA, "Upload checksum mismatch"
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        path = pathlib.PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute() and ".." not in path.parts
    tar.extractall(root)
manifest = json.loads((root / "deployment-manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
python = pathlib.Path.home() / "MLSys-HW/.venv/bin/python"
assert python.is_file(), "Existing server Python missing"
script = root / "homework-3/scripts/server_smoke.py"
with (root / "suite.log").open("xb") as log:
    process = subprocess.Popen(["bash", "-c", 'source /usr/local/Ascend/cann-9.0.0/set_env.sh && exec "$@"',
        "hw3", str(python), "-u", str(script), "--device", str(DEVICE), "--suite", SUITE,
        "--tp", str(TP), "--model-size", MODEL_SIZE] + (["--after-run", AFTER_RUN] if AFTER_RUN else []),
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env)
(root / "launch.json").write_text(json.dumps({"pid": process.pid, "root": str(root)}) + "\\n")
print(json.dumps({"pid": process.pid, "root": str(root)}), flush=True)
'''
    code = f"LABEL={label!r}\nSHA={sha!r}\nDEVICE={args.device!r}\nSUITE={args.suite!r}\nTP={args.tp!r}\nMODEL_SIZE={args.model_size!r}\nAFTER_RUN={args.after_run!r}\n" + bootstrap
    result = subprocess.run([*ssh, "python3 -"], input=code, text=True,
                            capture_output=True, timeout=45)
    (local / "launch-output.txt").write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Launch was not confirmed. Inspect {remote}/suite.log before retrying.")
    print("Started independently of this Terminal/chat. Full HW3 validation is NOT complete.")
    print("Check status with:\n" + shlex.join([*ssh, f"cat {remote}/status.json; tail -n 20 {remote}/suite.log"]))
    print("Retrieve later with:\n" + shlex.join(["scp", "-r", "-P", "2222", *common,
          f"{args.host}:{remote}/results", str(local / "results")]))


if __name__ == "__main__":
    main()
