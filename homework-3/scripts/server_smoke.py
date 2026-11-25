"""Course-server 0.8B TP1 smoke only; no claim of complete offload validation."""
import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path.home() / "MLSys-HW"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_status(**values):
    values["updated_unix"] = time.time()
    tmp = ROOT / "status.tmp"
    tmp.write_text(json.dumps(values, indent=2) + "\n")
    tmp.replace(ROOT / "status.json")
    print(json.dumps(values), flush=True)


def bounded(command, name, seconds=600):
    """Only terminate this invocation's process group, including leftover workers."""
    with (ROOT / "results" / f"{name}.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                time.sleep(2)
                # Reap an exited leader before probing/killing remaining workers.
                # macOS can return EPERM when a group contains only a zombie.
                process.poll()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect results/{name}.log")


def prepare_source():
    helper = load("hw2_smoke", ROOT / "homework-2/scripts/run_model.py")
    manifest = json.loads((ROOT / "deployment-manifest.json").read_text())
    previous = BASE / "hw2-streaming-20261109/homework-1/tokenspeed"
    source = ROOT / "homework-1/tokenspeed"
    if not previous.is_dir():
        raise RuntimeError(f"Previously deployed HW2 source missing: {previous}")
    source.parent.mkdir()
    shutil.copytree(previous, source, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    for relative, expected in manifest["expected_runtime_trees"].items():
        actual = helper.fingerprint_python_tree(source / relative)
        if actual["files"] != expected["files"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Server source differs from reviewed source: {relative}")
    os.environ["PYTHONPATH"] = os.pathsep.join(str(source / name) for name in manifest["expected_runtime_trees"])
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def ports_available(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("0.0.0.0", port))
        return True
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        return False
    finally:
        for sock in sockets:
            sock.close()


def choose_ports():
    for engine in range(31893, 31913):
        # Prefer ranges below Linux's usual ephemeral-client range: scattered
        # outgoing connections can occupy every candidate 32-port block above it.
        for start in range(25000, 28000, 64):
            if ports_available([engine, *range(start, start + 32)]):
                os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{start}-{start + 31}"
                return engine, list(range(start, start + 32))
    raise RuntimeError("No available engine/HCCL port range; no existing process was modified")


def physical_devices(device, tp):
    if tp not in (1, 2) or device not in range(8) or (tp == 2 and device % 2):
        raise ValueError("TP2 requires an even first physical chip on the recorded course topology")
    return list(range(device, device + tp))


def experiment_jobs(suite):
    jobs = []
    patterns = ("high", "low") if suite == "matrix" else ("high",)
    for pattern in patterns:
        for repetition in range(3 if suite == "matrix" else 1):
            prefix = f"{pattern}-r{repetition}-" if suite == "matrix" else ""
            cases = [("off", "off", "0", "0")]
            if suite in ("matrix", "capacity"):
                small, large = ("0.5", "1") if suite == "matrix" else ("1", "8")
                cases += [("on-small", "on", small, "0"), ("on-large", "on", large, "0")]
            else:
                cases += [("on-decode", "on", "1", "2"), ("on-prefill", "on", "1", "-2")]
            for name, mode, ratio, force in cases:
                jobs.append(dict(name=prefix + name, mode=mode, ratio=ratio, force=force,
                                 pattern=pattern, reference=prefix + "off", repetition=repetition,
                                 working_set=8 if suite in ("matrix", "stress", "capacity") else 4,
                                 cycles=8 if suite == "stress" else 2))
    return jobs


def assert_free(device, ports=(), tp=1):
    devices = physical_devices(device, tp)
    result = subprocess.run(["npu-smi", "info"], check=True, capture_output=True, text=True, timeout=15)
    occupancy = load("smoke_occupancy", ROOT / "homework-2/scripts/npu_occupancy.py")
    occupied = occupancy.occupied_chips(result.stdout)
    if any((d // 2, d % 2) in occupied for d in devices):
        raise RuntimeError(f"Requested physical chips {devices} include an occupied chip; no job started")
    if not ports_available(ports):
        raise RuntimeError("A selected port became occupied; no existing process was modified")
    return result.stdout


def wait_for_run(label, timeout=1800):
    """Wait for a named successful predecessor without holding the device lock."""
    if not re.fullmatch(r"hw3-smoke-\d{8}T\d{6}Z-[a-f0-9]{8}", label):
        raise ValueError("invalid predecessor label")
    deadline = time.monotonic() + timeout
    while True:
        path = BASE / label / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        state = status.get("status")
        if state in ("suite_passed", "smoke_passed"):
            return
        if state in ("failed", "blocked"):
            raise RuntimeError(f"Predecessor {label} ended {state}; queued experiment cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Predecessor {label} did not finish within {timeout}s")
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, choices=range(8), required=True)
    parser.add_argument("--suite", choices=("smoke", "trace", "matrix", "stress", "capacity"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model-size", choices=("0.8B", "4B"), required=True)
    parser.add_argument("--after-run", help="Optional successful predecessor; wait at most 30 minutes")
    args = parser.parse_args()
    devices = physical_devices(args.device, args.tp)
    traced = args.suite != "smoke"
    if args.after_run:
        save_status(status="waiting", predecessor=args.after_run)
        try:
            wait_for_run(args.after_run)
        except Exception as exc:
            save_status(status="blocked", error=f"{type(exc).__name__}: {exc}")
            return 1
    # Coordinates our launchers only; this cannot reserve a shared-server chip.
    with (BASE / ".hw3-smoke.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            save_status(status="blocked", reason="Another HW3 smoke launcher is already active")
            return 1
        try:
            (ROOT / "results").mkdir()
            save_status(status="running", stage="preflight", device=args.device)
            prepare_source()
            if traced:
                os.environ["PYTHONPATH"] = str(ROOT / "homework-3/runtime") + os.pathsep + os.environ["PYTHONPATH"]
            model = BASE / "models" / f"Qwen3.5-{args.model_size}"
            config = ROOT / "homework-2/metadata" / f"Qwen3.5-{args.model_size}" / "config.json"
            if (model / "config.json").read_bytes() != config.read_bytes():
                raise RuntimeError("Server model configuration differs from prepared configuration")
            # Check model artifacts against the existing download receipt before NPU use.
            receipt = json.loads((model / "source-manifest.json").read_text())
            for entry in receipt["files"]:
                path = model / entry["path"]
                if not path.resolve().is_relative_to(model.resolve()):
                    raise ValueError("Model receipt path escapes model directory")
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"Model file checksum mismatch: {entry['path']}")
            bounded([sys.executable, "-m", "unittest", "discover", "-s", "homework-3/tests", "-v"],
                    "unit-tests", 180)
            port, hccl_ports = choose_ports()
            (ROOT / "results/ports.json").write_text(json.dumps({"engine": port, "hccl": hccl_ports}) + "\n")
            common = [sys.executable, "homework-3/scripts/run_offloading.py", "--action", "run",
                      "--model", str(model), "--model-config", str(config), "--seed-cases", "homework-2/cases.json",
                      "--tp", str(args.tp), "--physical-devices", ",".join(map(str, devices)),
                      "--memory-fraction", "0.12" if args.model_size == "0.8B" else "0.25",
                      "--device-tokens", "16384", "--port", str(port),
                      "--dtype", "bfloat16" if args.model_size == "0.8B" else "float16",
                      "--head-dtype", "native" if args.model_size == "0.8B" else "float32",
                      "--revisit-suffix-tokens", "8"]
            jobs = experiment_jobs(args.suite)
            (ROOT / "results/experiment-plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
            for job in jobs:
                name, mode, ratio, force = (job[k] for k in ("name", "mode", "ratio", "force"))
                snapshot = assert_free(args.device, [port, *hccl_ports], args.tp)
                (ROOT / "results" / f"{name}-preflight.txt").write_text(snapshot)
                save_status(status="running", stage=name, device=args.device)
                if traced:
                    os.environ["HW3_TRACE_DIR"] = str(ROOT / "results" / name / "trace")
                bounded([*common, "--offloading", mode, "--host-ratio", ratio,
                         "--snapshot-ratio", "0" if mode == "off" else "1", "--force-interval", force,
                         "--pattern", job["pattern"], "--repeats", str(job["cycles"]),
                         "--working-set", str(job["working_set"]),
                         "--output-dir", f"results/{name}"], name)
                if traced:
                    bounded([sys.executable, "homework-3/scripts/summarize_trace.py", "--run-dir", f"results/{name}"],
                            f"trace-{name}", 60)
                if mode == "on":
                    bounded([sys.executable, "homework-3/scripts/compare_results.py",
                             "--reference", f"results/{job['reference']}/result.json", "--candidate", f"results/{name}/result.json",
                             "--output", f"results/{name}-comparison.json"], f"compare-{name}", 60)
            save_status(status="suite_passed", suite=args.suite, model_size=args.model_size, tp=args.tp,
                        jobs_completed=len(jobs), offload_lifecycle_verified=False,
                        limitation="Individual suite completed; full assignment acceptance requires evidence across all requirements")
            return 0
        except Exception as exc:
            save_status(status="failed", error=f"{type(exc).__name__}: {exc}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
