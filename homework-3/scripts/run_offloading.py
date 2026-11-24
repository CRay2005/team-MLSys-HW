"""HW3 paired workload runner. Generation agreement alone is not offload proof."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "homework-2/scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hw2 = load_helper("hw2_runner", "run_model.py")
occupancy = load_helper("hw2_occupancy", "npu_occupancy.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_config(model, *, tp, devices, memory_fraction, port, offloading,
                 host_ratio, snapshot_ratio, force_interval, device_tokens, dtype,
                 head_dtype):
    """Use the existing scheduler/L2 path; explicitly size both Host pools."""
    if offloading not in ("off", "on"):
        raise ValueError("offloading must be off or on")
    if dtype not in ("bfloat16", "float16") or head_dtype not in ("native", "float32"):
        raise ValueError("unsupported precision")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (host_ratio, snapshot_ratio)):
        raise ValueError("Host ratios must be finite and nonnegative")
    if type(force_interval) is not int:
        raise ValueError("force interval must be an integer")
    if type(device_tokens) is not int or device_tokens < 512:
        raise ValueError("device token budget must be an integer >= 512")
    if offloading == "off" and (host_ratio or snapshot_ratio or force_interval):
        raise ValueError("off baseline must disable BOTH Host pools and forced retraction")
    if offloading == "on" and (host_ratio <= 0 or snapshot_ratio <= 0):
        raise ValueError("on run requires explicit positive L2 and snapshot ratios")
    config = hw2.build_config(model, tp=tp, devices=devices,
                             memory_fraction=memory_fraction, concurrency=2, port=port)
    config.update(dtype=dtype, kv_cache_dtype=dtype,
                  max_total_tokens=device_tokens, enable_prefix_caching=True,
                  disable_kvstore=offloading == "off", kvstore_io_backend="direct",
                  kvstore_ratio=host_ratio, kvstore_size=0,
                  retraction_snapshot_host_gb=0.0,
                  retraction_snapshot_ratio=snapshot_ratio,
                  retraction_snapshot_max_requests=0,
                  debug_force_retraction_interval=force_interval)
    if head_dtype == "float32":
        config["lm_head_compute_dtype"] = "float32"
    return config


def make_workload(seed_ids, *, vocab_size, pattern, repeats):
    """Synthetic valid-token stress cases; different first tokens isolate prefixes."""
    if (type(vocab_size) is not int or vocab_size < 16 or not seed_ids
            or len(seed_ids) + 16 > 512
            or any(type(i) is not int or not 0 <= i < vocab_size for i in seed_ids)):
        raise ValueError("invalid seed tokens or vocabulary")
    if pattern not in ("high", "low") or type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("invalid pattern or repeats")
    # Both patterns issue eight requests per repeat with the same lengths.
    # High revisits two prefixes; low never revisits a prefix in this run.
    if vocab_size < 8 * repeats:
        raise ValueError("vocabulary too small for distinct low-reuse markers")
    batches = []
    for repeat in range(repeats):
        for batch in range(4):
            sessions = [0, 1] if pattern == "high" else [repeat * 8 + batch * 2 + i for i in range(2)]
            if pattern == "high" and batch % 2:
                sessions.reverse()
            batches.append({"repeat": repeat, "batch": batch,
                            "sessions": sessions,
                            "input_ids": [[(seed_ids[0] + s) % vocab_size, *seed_ids[1:]]
                                          for s in sessions]})
    return {"pattern": pattern, "new_tokens": 16, "batches": batches,
            "input_kind": "synthetic token stress workload; not a language-quality benchmark"}


def validate_outputs(outputs, batch, new_tokens):
    if not isinstance(outputs, list) or len(outputs) != len(batch):
        raise ValueError("missing request output")
    for output, ids in zip(outputs, batch):
        tokens, meta = output["output_ids"], output["meta_info"]
        if (len(tokens) != new_tokens or any(type(t) is not int or t < 0 for t in tokens)
                or meta["completion_tokens"] != new_tokens
                or meta["prompt_tokens"] != len(ids)):
            raise ValueError("incomplete or malformed generation")
        cached = meta["cached_tokens"]
        if type(cached) is not int or not 0 <= cached <= len(ids):
            raise ValueError("invalid cached-token count")


def compare_runs(reference, candidate):
    """Check every token under matched conditions; do not infer a Host hit."""
    for result in (reference, candidate):
        if result["status"] != "generation_completed_unverified":
            raise ValueError("both runs must complete")
        if result["workload_sha256"] != digest(result["workload"]):
            raise ValueError("workload digest mismatch")
        if len(result["runs"]) != len(result["workload"]["batches"]):
            raise ValueError("incomplete workload")
    if reference["config"]["disable_kvstore"] is not True or candidate["config"]["disable_kvstore"] is not False:
        raise ValueError("compare off reference against on candidate")
    if reference["config"]["retraction_snapshot_ratio"] != 0:
        raise ValueError("reference still has a Host snapshot pool")
    allowed = {"disable_kvstore", "kvstore_ratio", "retraction_snapshot_ratio",
               "debug_force_retraction_interval", "port"}
    if ({k: v for k, v in reference["config"].items() if k not in allowed}
            != {k: v for k, v in candidate["config"].items() if k not in allowed}):
        raise ValueError("unmatched device budget/model/precision/configuration")
    for key in ("workload_sha256", "model_config_sha256", "model_manifest_sha256",
                "source_fingerprints", "runtime_versions", "physical_devices"):
        if reference[key] != candidate[key]:
            raise ValueError(f"unmatched {key}")
    checked = 0
    for a, b, batch in zip(reference["runs"], candidate["runs"], reference["workload"]["batches"]):
        for row in (a, b):
            if row["repeat"] != batch["repeat"] or row["batch"] != batch["batch"]:
                raise ValueError("request ordering differs")
            validate_outputs(row["outputs"], batch["input_ids"], reference["workload"]["new_tokens"])
        if [o["output_ids"] for o in a["outputs"]] != [o["output_ids"] for o in b["outputs"]]:
            raise ValueError("generated token IDs differ")
        checked += len(a["outputs"])
    return {"exact_token_agreement_requests": checked,
            "offload_lifecycle_verified": False,
            "reason": "Requires per-rank copy, restore-position, allocation/reuse and eviction evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--seed-cases", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--physical-devices", type=hw2.parse_devices, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--device-tokens", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--offloading", choices=("off", "on"), required=True)
    parser.add_argument("--host-ratio", type=float, required=True)
    parser.add_argument("--snapshot-ratio", type=float, required=True)
    parser.add_argument("--force-interval", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--head-dtype", choices=("native", "float32"), required=True)
    parser.add_argument("--pattern", choices=("high", "low"), required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = build_config(args.model, tp=args.tp, devices=args.physical_devices,
                          memory_fraction=args.memory_fraction, port=args.port,
                          offloading=args.offloading, host_ratio=args.host_ratio,
                          snapshot_ratio=args.snapshot_ratio, force_interval=args.force_interval,
                          device_tokens=args.device_tokens, dtype=args.dtype, head_dtype=args.head_dtype)
    config_bytes = args.model_config.read_bytes()
    vocab = json.loads(config_bytes)["text_config"]["vocab_size"]
    seed = next(c["input_ids"] for c in json.loads(args.seed_cases.read_text())["cases"]
                if len(c["input_ids"]) == 256)
    workload = make_workload(seed, vocab_size=vocab, pattern=args.pattern, repeats=args.repeats)
    result = {"status": "planned_not_run", "config": config, "workload": workload,
              "physical_devices": args.physical_devices, "workload_sha256": digest(workload),
              "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(), "runs": [],
              "missing_evidence": ["per-rank device block use/free/reallocation", "Host pool use/evictions",
                                   "D2H/H2D bytes and completed-transfer timing", "restore context agreement",
                                   "Host hit/miss and replay ranges/timing", "steady-state leak checks"],
              "measurement": "End-to-end batch latency, including replay; not isolated restore latency."}
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def save():
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.action == "plan":
        return
    engine = None
    try:
        manifest_bytes = (args.model / "source-manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        expected = next(f["sha256"] for f in manifest["files"] if f["path"] == "config.json")
        if hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest() != expected or expected != result["model_config_sha256"]:
            raise ValueError("model config/manifest mismatch")
        result["model_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        result["source_fingerprints"] = hw2.project_source_fingerprints()
        if any("error" in f for f in result["source_fingerprints"].values()):
            raise ValueError("source tree unavailable for fingerprinting")
        hw2.snapshot(args.output_dir / "npu-before.txt")
        used = occupancy.occupied_chips((args.output_dir / "npu-before.txt").read_text())
        devices = args.physical_devices
        # Course server's previously recorded 4-card/2-chip mapping. Reverify
        # topology manually before use on a different machine; this is no lock.
        if any(d not in range(8) or (d // 2, d % 2) in used for d in devices):
            raise RuntimeError("requested chip occupied or outside recorded topology")
        if args.tp == 2 and devices[0] // 2 != devices[1] // 2:
            raise ValueError("TP2 requires the recorded same-card pair")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed.runtime.entrypoints.engine import Engine
        if not torch.npu.is_available() or torch.npu.device_count() != args.tp:
            raise RuntimeError("NPU availability/count mismatch")
        result["runtime_versions"] = hw2.runtime_versions()
        result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["environment"] = hw2.selected_environment()
        engine = Engine(**config)
        result["status"] = "running"
        result["process_snapshot"] = hw2.process_snapshot()
        hw2.snapshot(args.output_dir / "npu-loaded.txt")
        save()
        for batch in workload["batches"]:
            start = time.perf_counter()
            outputs = engine.generate(input_ids=batch["input_ids"], sampling_params={
                "temperature": 0, "max_new_tokens": workload["new_tokens"], "ignore_eos": True})
            elapsed = time.perf_counter() - start
            validate_outputs(outputs, batch["input_ids"], workload["new_tokens"])
            result["runs"].append({"repeat": batch["repeat"], "batch": batch["batch"],
                                   "seconds": elapsed, "outputs": outputs,
                                   "output_tokens_per_second": 2 * workload["new_tokens"] / elapsed})
            save()
        hw2.snapshot(args.output_dir / "npu-after-workload.txt")
        result["status"] = "generation_completed_unverified"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        except BaseException as exc:
            result.update(status="failed", shutdown_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
