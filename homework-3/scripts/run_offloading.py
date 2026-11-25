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
SOURCE_TREES = ("python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python")
HARNESS_FILES = ("homework-3/scripts/run_offloading.py", "homework-2/scripts/run_model.py",
                 "homework-2/scripts/npu_occupancy.py")


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "homework-2/scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hw2 = load_helper("hw2_runner", "run_model.py")
occupancy = load_helper("hw2_occupancy", "npu_occupancy.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def harness_fingerprints():
    """Bind measurements to this runner and the HW2 code it actually imports."""
    paths = tuple(ROOT / name for name in HARNESS_FILES)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def observer_fingerprints():
    enabled = bool(os.environ.get("HW3_TRACE_DIR"))
    paths = [ROOT / "homework-3/runtime/sitecustomize.py", ROOT / "homework-3/runtime/runtime_observer.py",
             ROOT / "homework-3/scripts/summarize_trace.py"] if enabled else []
    return {"enabled": enabled, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def validate_source_locations(locations):
    """Reject installed packages outside the source trees recorded in the result."""
    base = ROOT / "homework-1/tokenspeed"
    expected = {"tokenspeed": base / "python/tokenspeed",
                "tokenspeed_kernel": base / "tokenspeed-kernel/python/tokenspeed_kernel",
                "tokenspeed_kernel_npu": base / "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu"}
    for name, directory in expected.items():
        if (name not in locations or len(locations[name]) != 1
                or Path(locations[name][0]).resolve() != directory.resolve()):
            raise ValueError(f"runtime source location differs from fingerprinted tree: {name}")


def save_result(path, result):
    """Replace a complete JSON record; interruption cannot truncate the last save."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_devices(tp, devices):
    """Validate the previously recorded course-server topology, even for plans."""
    if (type(tp) is not int or tp not in (1, 2) or not isinstance(devices, list)
            or len(devices) != tp or any(type(d) is not int or d not in range(8) for d in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("expected one/two distinct devices in the recorded 0..7 topology")
    if tp == 2 and devices[0] // 2 != devices[1] // 2:
        raise ValueError("TP2 requires the recorded same-card pair")


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_provenance(result):
    """Equal missing evidence is not evidence; validate the recorded schema first."""
    for key in ("model_config_sha256", "model_manifest_sha256"):
        if not valid_sha256(result.get(key)):
            raise ValueError(f"invalid {key}")
    if "observer" in result:
        observer = result["observer"]
        if (not isinstance(observer, dict) or set(observer) != {"enabled", "files"}
                or type(observer["enabled"]) is not bool or not isinstance(observer["files"], dict)
                or set(observer["files"]) != ({"sitecustomize.py", "runtime_observer.py", "summarize_trace.py"} if observer["enabled"] else set())
                or not all(valid_sha256(v) for v in observer["files"].values())):
            raise ValueError("invalid observer provenance")
    harness = result.get("harness_fingerprints")
    if (not isinstance(harness, dict) or set(harness) != set(HARNESS_FILES)
            or not all(valid_sha256(v) for v in harness.values())):
        raise ValueError("incomplete harness_fingerprints")
    trees = result.get("source_fingerprints")
    if not isinstance(trees, dict) or set(trees) != set(SOURCE_TREES):
        raise ValueError("incomplete source_fingerprints")
    for name, tree in trees.items():
        if (not isinstance(tree, dict) or not isinstance(tree.get("root"), str) or not tree["root"]
                or not valid_sha256(tree.get("sha256")) or "error" in tree):
            raise ValueError(f"invalid source_fingerprints: {name}")
        files = tree.get("files")
        if (not isinstance(files, dict) or not files
                or any(not isinstance(p, str) or not p or not valid_sha256(sha)
                       for p, sha in files.items())):
            raise ValueError(f"incomplete source file hashes: {name}")
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != tree["sha256"]:
            raise ValueError(f"source tree digest mismatch: {name}")
    versions = result.get("runtime_versions")
    if (not isinstance(versions, dict) or set(versions) != {"torch", "torch_npu", "triton"}
            or any(not isinstance(v, str) or not v.strip() for v in versions.values())
            or any(versions[k] == "unavailable" for k in ("torch", "torch_npu"))):
        raise ValueError("incomplete runtime_versions")
    validate_devices(result["config"].get("world_size"), result.get("physical_devices"))


def build_config(model, *, tp, devices, memory_fraction, port, offloading,
                 host_ratio, snapshot_ratio, force_interval, device_tokens, dtype,
                 head_dtype):
    """Use the existing scheduler/L2 path; explicitly size both Host pools."""
    validate_devices(tp, devices)
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


def make_workload(seed_ids, *, vocab_size, pattern, repeats, working_set, revisit_suffix_tokens):
    """Synthetic valid-token stress cases; different first tokens isolate prefixes."""
    if (type(vocab_size) is not int or vocab_size < 16 or not seed_ids
            or len(seed_ids) + 16 > 512
            or any(type(i) is not int or not 0 <= i < vocab_size for i in seed_ids)):
        raise ValueError("invalid seed tokens or vocabulary")
    if pattern not in ("high", "low") or type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("invalid pattern or repeats")
    if type(working_set) is not int or not 2 <= working_set <= 64 or working_set % 2:
        raise ValueError("working set must be an even integer in 2..64")
    if (type(revisit_suffix_tokens) is not int or revisit_suffix_tokens < 0
            or len(seed_ids) + revisit_suffix_tokens + 16 > 512):
        raise ValueError("invalid revisit suffix or context length")
    # Same request counts and lengths across patterns; low never revisits a prefix.
    if vocab_size < working_set * repeats:
        raise ValueError("vocabulary too small for distinct low-reuse markers")
    batches = []
    for repeat in range(repeats):
        suffix = ([seed_ids[i % len(seed_ids)] for i in range(revisit_suffix_tokens)]
                  if repeat else [])
        for batch in range(working_set // 2):
            sessions = ([(repeat * 2 + batch * 2 + i) % working_set for i in range(2)]
                        if pattern == "high" else [repeat * working_set + batch * 2 + i for i in range(2)])
            if (batch + repeat) % 2:
                sessions.reverse()
            batches.append({"repeat": repeat, "batch": batch,
                            "phase": "initial" if repeat == 0 else "extended",
                            "sessions": sessions,
                            "input_ids": [[(seed_ids[0] + s) % vocab_size, *seed_ids[1:], *suffix]
                                          for s in sessions]})
    return {"pattern": pattern, "new_tokens": 16, "batches": batches,
            "working_set": working_set, "repeats": repeats, "revisit_suffix_tokens": revisit_suffix_tokens,
            "input_kind": "synthetic token stress workload; not a language-quality benchmark"}


def validate_outputs(outputs, batch, new_tokens):
    if not isinstance(outputs, list) or len(outputs) != len(batch):
        raise ValueError("missing request output")
    for output, ids in zip(outputs, batch):
        tokens, meta = output["output_ids"], output["meta_info"]
        if (not isinstance(tokens, list) or len(tokens) != new_tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or type(meta["completion_tokens"]) is not int
                or type(meta["prompt_tokens"]) is not int
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
        validate_provenance(result)
        workload = result["workload"]
        if (not isinstance(workload["batches"], list) or not workload["batches"]
                or type(workload["new_tokens"]) is not int or workload["new_tokens"] <= 0):
            raise ValueError("empty or malformed workload")
        identifiers = []
        for batch in workload["batches"]:
            if (type(batch["repeat"]) is not int or batch["repeat"] < 0
                    or type(batch["batch"]) is not int or batch["batch"] < 0
                    or len(batch["input_ids"]) != 2
                    or any(not ids or any(type(i) is not int or i < 0 for i in ids)
                           for ids in batch["input_ids"])):
                raise ValueError("malformed workload batch")
            identifiers.append((batch["repeat"], batch["batch"]))
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate workload batch")
        if result["workload_sha256"] != digest(result["workload"]):
            raise ValueError("workload digest mismatch")
        if len(result["runs"]) != len(result["workload"]["batches"]):
            raise ValueError("incomplete workload")
    if reference["config"]["disable_kvstore"] is not True or candidate["config"]["disable_kvstore"] is not False:
        raise ValueError("compare off reference against on candidate")
    for key in ("kvstore_ratio", "kvstore_size", "retraction_snapshot_host_gb",
                "retraction_snapshot_ratio", "retraction_snapshot_max_requests",
                "debug_force_retraction_interval"):
        value = reference["config"].get(key)
        if type(value) not in (int, float) or value != 0:
            raise ValueError(f"off reference must disable {key}")
    for key in ("kvstore_ratio", "retraction_snapshot_ratio"):
        value = candidate["config"].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"on candidate needs a finite positive {key}")
    # Negative intervals deliberately force prefill suspension; positive ones decode.
    if type(candidate["config"].get("debug_force_retraction_interval")) is not int:
        raise ValueError("candidate retraction interval must be an integer")
    allowed = {"disable_kvstore", "kvstore_ratio", "retraction_snapshot_ratio",
               "debug_force_retraction_interval", "port"}
    if ({k: v for k, v in reference["config"].items() if k not in allowed}
            != {k: v for k, v in candidate["config"].items() if k not in allowed}):
        raise ValueError("unmatched device budget/model/precision/configuration")
    for key in ("workload_sha256", "model_config_sha256", "model_manifest_sha256",
                "source_fingerprints", "harness_fingerprints", "runtime_versions", "physical_devices"):
        if reference[key] != candidate[key]:
            raise ValueError(f"unmatched {key}")
    if reference.get("observer") != candidate.get("observer"):
        raise ValueError("unmatched observer instrumentation")
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
    parser.add_argument("--working-set", type=int, required=True)
    parser.add_argument("--revisit-suffix-tokens", type=int, required=True)
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
    workload = make_workload(seed, vocab_size=vocab, pattern=args.pattern, repeats=args.repeats,
                             working_set=args.working_set, revisit_suffix_tokens=args.revisit_suffix_tokens)
    result = {"status": "planned_not_run", "config": config, "workload": workload,
              "harness_fingerprints": harness_fingerprints(),
              "observer": observer_fingerprints(),
              "physical_devices": args.physical_devices, "workload_sha256": digest(workload),
              "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(), "runs": [],
              "missing_evidence": ["per-rank device block use/free/reallocation", "Host pool use/evictions",
                                   "D2H/H2D bytes and completed-transfer timing", "restore context agreement",
                                   "Host hit/miss and replay ranges/timing", "steady-state leak checks"],
              "measurement": "End-to-end batch latency, including replay; not isolated restore latency."}
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def save():
        save_result(args.output_dir / "result.json", result)

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
        if any((d // 2, d % 2) in used for d in devices):
            raise RuntimeError("requested chip occupied")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import torch_npu  # noqa: F401
        locations = {}
        for name in ("tokenspeed", "tokenspeed_kernel", "tokenspeed_kernel_npu"):
            spec = importlib.util.find_spec(name)
            locations[name] = list(spec.submodule_search_locations or ()) if spec else []
        validate_source_locations(locations)
        result["runtime_source_locations"] = locations
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
        if result["observer"]["enabled"]:
            from summarize_trace import wait_for_quiescence
            wait_for_quiescence(os.environ["HW3_TRACE_DIR"], args.tp)
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
