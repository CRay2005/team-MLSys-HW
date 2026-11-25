"""Run HW3 on Ascend and compare matched Host-cache trials."""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "homework-2" / "scripts"))
import run_model as hw2
from summarize_large_run import owned_hbm

COMPLETED = "generation_completed_not_yet_compared"
TRACE_TAGS = ("[host-cache-pools]", "[host-cache-round]", "[host-cache-copy]")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def workload(pattern, prompt_length, rounds, output_tokens):
    """Disjoint prefixes with deterministic suffixes and equal work across cache modes."""
    rng = random.Random(3)
    prefixes = {}

    def prompt(session):
        if session not in prefixes:
            # The first token makes keys of independent sessions disjoint.
            prefixes[session] = [1000 + session] + [
                rng.randrange(2048, 20000) for _ in range(prompt_length - 1)
            ]
        return prefixes[session]

    if pattern == "retract":
        return [
            ([prompt(2 * i), prompt(2 * i + 1)], output_tokens) for i in range(rounds)
        ]

    trace = [([prompt(0)], 1), ([prompt(1)], 1)]
    for i in range(rounds):
        trace.append(([prompt(i + 2)], 1))
        session = i % 2 if pattern == "high" else rounds + i + 2
        trace.append(([prompt(session) + [20100 + i]], output_tokens))
    return trace


def run(args):
    require(args.rounds in range(1, 101), "rounds must be in 1..100")
    require(32 <= args.prompt_length <= 447, "prompt length must be in 32..447")
    require(4 <= args.output_tokens <= 64, "output tokens must be in 4..64")
    require(args.device_tokens > 0, "device tokens must be positive")
    require(
        math.isfinite(args.host_ratio) and args.host_ratio >= 0, "invalid Host ratio"
    )
    require(
        math.isfinite(args.snapshot_gb) and args.snapshot_gb >= 0,
        "invalid snapshot budget",
    )
    require(args.force_interval >= 0, "invalid forced retraction interval")
    require(
        not args.force_interval or (args.pattern == "retract" and args.snapshot_gb > 0),
        "forced retraction requires the retract pattern and a snapshot budget",
    )
    config = hw2.build_config(
        args.model,
        tp=args.tp,
        devices=args.physical_devices,
        memory_fraction=args.memory_fraction,
        concurrency=2,
        port=args.port,
    )
    config.update(
        dtype=args.dtype,
        max_total_tokens=args.device_tokens,
        enable_prefix_caching=True,
        disable_kvstore=args.host_ratio == 0,
        kvstore_ratio=args.host_ratio or 2.0,
        kvstore_io_backend="direct",
        retraction_snapshot_host_gb=args.snapshot_gb,
        retraction_snapshot_ratio=None if args.snapshot_gb else 0.0,
        debug_force_retraction_interval=args.force_interval,
    )
    if args.dtype == "float16":
        config["kv_cache_dtype"] = "float16"
    if args.lm_head_compute_dtype:
        config["lm_head_compute_dtype"] = args.lm_head_compute_dtype
    model_config = (args.model / "config.json").read_bytes()
    manifest = (args.model / "source-manifest.json").read_bytes()
    config_hash = hashlib.sha256(model_config).hexdigest()
    require(
        any(
            entry["path"] == "config.json" and entry["sha256"] == config_hash
            for entry in json.loads(manifest)["files"]
        ),
        "model config does not match its source manifest",
    )
    trace = workload(args.pattern, args.prompt_length, args.rounds, args.output_tokens)
    vocab = json.loads(model_config)["text_config"]["vocab_size"]
    require(
        all(max(prompt) < vocab for batch, _ in trace for prompt in batch),
        "workload token outside model vocabulary",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TOKENSPEED_TRACE_HOST_CACHE"] = "1"
    result = {
        "status": "starting",
        "tp": args.tp,
        "physical_devices": args.physical_devices,
        "config": config,
        "pattern": args.pattern,
        "config_sha256": config_hash,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "patch_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "patches"
                / "ascend-host-cache.patch"
            ).read_bytes()
        ).hexdigest(),
        "environment": hw2.selected_environment(),
        "runs": [],
    }
    result_file = args.output_dir / "result.json"
    save(result_file, result)
    engine = None
    try:
        import torch
        import torch_npu  # noqa: F401

        require(
            torch.npu.is_available() and torch.npu.device_count() == args.tp,
            "NPU availability or visible device count mismatch",
        )
        from tokenspeed.runtime.entrypoints.engine import Engine

        result["runtime_versions"] = hw2.runtime_versions()
        hw2.snapshot(args.output_dir / "hbm-before.txt")
        engine = Engine(**config)
        result["process_snapshot_after_load"] = hw2.process_snapshot()
        hw2.snapshot(args.output_dir / "hbm-loaded.txt")
        for index, (batch, count) in enumerate(trace):
            print(f"[hw3-run] index={index}", flush=True)
            params = {"temperature": 0, "max_new_tokens": count, "ignore_eos": True}
            start = time.perf_counter()
            outputs = engine.generate(
                input_ids=batch, sampling_params=params, return_logprob=True
            )
            elapsed = time.perf_counter() - start
            require(len(outputs) == len(batch), f"batch {index} incomplete")
            rows = []
            for prompt, output in zip(batch, outputs):
                meta = output["meta_info"]
                ids = output["output_ids"]
                logs = meta["output_token_logprobs"]
                require(
                    meta["prompt_tokens"] == len(prompt)
                    and len(ids) == len(logs) == count,
                    f"batch {index} truncated",
                )
                require(
                    [entry[1] for entry in logs] == ids
                    and all(math.isfinite(entry[0]) for entry in logs),
                    f"batch {index} invalid logprobs",
                )
                rows.append(
                    {
                        "prompt": prompt,
                        "tokens": ids,
                        "logprobs": [entry[0] for entry in logs],
                        "cached_tokens": meta["cached_tokens"],
                    }
                )
            result["runs"].append({"wall_seconds": elapsed, "rows": rows})
            save(result_file, result)
        hw2.snapshot(args.output_dir / "hbm-after.txt")
        result["status"] = COMPLETED
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            save(result_file, result)


def read_trace(path, tp):
    pools = {}
    by_rank = {rank: {"rounds": [], "copies": []} for rank in range(tp)}
    for line in path.read_text().splitlines():
        tag = next((tag for tag in TRACE_TAGS if tag in line), None)
        if tag is None:
            continue
        payload = line.split(tag, 1)[1]
        # TokenSpeed adds the source location and an ANSI color reset.
        payload = re.sub(r" \([^()]+:\d+\)(?:\x1b\[[0-9;]*m)?$", "", payload)
        suffix_key = {"[host-cache-round]": "prefill", "[host-cache-copy]": "ids"}.get(
            tag
        )
        fields, _, suffix = (
            payload.partition(f" {suffix_key}=") if suffix_key else (payload, "", "")
        )
        row = dict(item.split("=", 1) for item in fields.split())
        rank = int(row.pop("rank"))
        require(rank in by_rank, f"unexpected rank {rank} in {path}")
        if tag == "[host-cache-pools]":
            require(rank not in pools, f"duplicate pool record in {path}")
            pools[rank] = {key: int(value) for key, value in row.items()}
        elif tag == "[host-cache-round]":
            round_stats = {key: int(value) for key, value in row.items()}
            round_stats["prefill"] = ast.literal_eval(suffix)
            by_rank[rank]["rounds"].append(round_stats)
        else:
            row.update(
                operations=int(row["operations"]),
                bytes=int(row["bytes"]),
                completion_ms=float(row["completion_ms"]),
                ids=ast.literal_eval(suffix),
            )
            by_rank[rank]["copies"].append(row)
    require(
        all(rank_data["rounds"] for rank_data in by_rank.values()),
        f"missing rounds in {path}",
    )
    return pools, by_rank


def summarize(directory, doc):
    pools, by_rank = read_trace(directory / "worker.log", doc["tp"])
    config = doc["config"]
    has_host = (
        not config["disable_kvstore"] or config["retraction_snapshot_host_gb"] > 0
    )
    require(
        set(pools) == (set(by_rank) if has_host else set()),
        "missing or unexpected Host pools",
    )
    per_rank = {}
    for rank, trace in by_rank.items():
        rounds = trace["rounds"]
        pool = pools.get(
            rank,
            {
                "l2_bytes": 0,
                "snapshot_bytes": 0,
                "blob_arena_bytes": 0,
                "lcm_block_bytes": 1,
            },
        )
        copies = trace["copies"]
        copy_summary = {}
        for kind in sorted({copy["kind"] for copy in copies}):
            group = [copy for copy in copies if copy["kind"] == kind]
            copy_summary[kind] = {
                "bytes": sum(copy["bytes"] for copy in group),
                "operations": sum(copy["operations"] for copy in group),
                "median_submission_to_completion_ms": median(
                    copy["completion_ms"] for copy in group
                ),
            }
        per_rank[rank] = {
            "pool": pool,
            "device_total_lcm_blocks": rounds[-1]["device_total"],
            "peak_used_device_lcm_blocks": max(row["device_used"] for row in rounds),
            "min_empty_device_lcm_blocks": min(row["device_empty"] for row in rounds),
            "min_reclaimable_device_lcm_blocks": min(
                row["device_reclaimable"] for row in rounds
            ),
            "final_reclaimable_device_lcm_blocks": rounds[-1]["device_reclaimable"],
            "final_empty_device_lcm_blocks": rounds[-1]["device_empty"],
            "peak_claimed_l2_bytes": pool["l2_bytes"]
            - min(row["host_free"] for row in rounds) * pool["lcm_block_bytes"],
            "peak_claimed_snapshot_bytes": pool["snapshot_bytes"]
            - min(row["snapshot_free"] for row in rounds) * pool["lcm_block_bytes"],
            "final_snapshot_free_lcm_blocks": rounds[-1]["snapshot_free"],
            "host_hits": rounds[-1]["host_hits"],
            "host_misses": rounds[-1]["host_misses"],
            "host_evictions": rounds[-1]["host_evictions"],
            "copies": copy_summary,
        }

    # Copy batches may complete in different orders across ranks. Require
    # matching operation IDs and context positions instead of ordering ACKs.
    def plan_signature(trace):
        positions = [tuple(row["prefill"]) for row in trace["rounds"] if row["prefill"]]
        operations = Counter(
            (copy["kind"], op_id) for copy in trace["copies"] for op_id in copy["ids"]
        )
        return positions, operations

    require(
        all(
            plan_signature(trace) == plan_signature(by_rank[0])
            for trace in by_rank.values()
        ),
        "TP ranks disagree on prefill positions or completed cache operations",
    )
    hbm = {
        stage: owned_hbm(doc, (directory / f"hbm-{stage}.txt").read_text())
        for stage in ("loaded", "after")
    }
    boundaries = Counter()
    if config["debug_force_retraction_interval"]:
        # Each mirrored scheduler prints the request and its context length.
        # Positive force intervals suspend decode with one sampled input pending.
        entries = Counter(
            re.findall(
                r"suspended request (\S+) \((\d+) tokens, computed=(\d+)\) with its image",
                (directory / "worker.log").read_text(),
            )
        )
        for (_, tokens, computed), count in entries.items():
            require(count % doc["tp"] == 0, "TP snapshot context positions differ")
            require(int(tokens) == int(computed) + 1, "unexpected decode boundary")
            boundaries[int(computed)] += count // doc["tp"]
        for data in per_rank.values():
            for kind in ("SnapshotDone", "RestoreDone"):
                require(
                    sum(boundaries.values()) == data["copies"][kind]["operations"],
                    "snapshot boundaries and completed copies differ",
                )
    # The high/low trace starts with two seeds, then one filler per measured
    # request. Keep all traffic in throughput and report target latency apart.
    runs = doc["runs"]
    seconds = sum(row["wall_seconds"] for row in runs)
    targets = runs if doc["pattern"] == "retract" else runs[3::2]
    # Exclude the first example of each request shape, including decode,
    # when reporting throughput after kernel warmup.
    warm_runs = runs[1:] if doc["pattern"] == "retract" else runs[4:]
    return {
        "median_target_batch_seconds": median(row["wall_seconds"] for row in targets),
        "output_tokens_per_second": sum(
            len(x["tokens"]) for row in runs for x in row["rows"]
        )
        / seconds,
        "post_warmup_output_tokens_per_second": (
            sum(len(x["tokens"]) for row in warm_runs for x in row["rows"])
            / sum(row["wall_seconds"] for row in warm_runs)
            if warm_runs
            else None
        ),
        "cached_prompt_tokens": sum(
            x["cached_tokens"] for row in targets for x in row["rows"]
        ),
        "snapshot_counts_by_computed_tokens": dict(sorted(boundaries.items())),
        "total_seconds": seconds,
        "rank": per_rank,
        "owned_hbm": hbm,
    }


def compare(args):
    baseline, trial = [
        json.loads((d / "result.json").read_text()) for d in (args.baseline, args.trial)
    ]
    require(
        baseline["status"] == trial["status"] == COMPLETED, "both runs must complete"
    )
    for key in (
        "tp",
        "physical_devices",
        "pattern",
        "config_sha256",
        "manifest_sha256",
        "patch_sha256",
        "runtime_versions",
    ):
        require(baseline[key] == trial[key], f"unmatched {key}")
    changed = {
        "disable_kvstore",
        "kvstore_ratio",
        "retraction_snapshot_host_gb",
        "retraction_snapshot_ratio",
        "debug_force_retraction_interval",
        "port",
    }
    require(
        {k: v for k, v in baseline["config"].items() if k not in changed}
        == {k: v for k, v in trial["config"].items() if k not in changed},
        "unmatched engine configs",
    )
    a, b = baseline["runs"], trial["runs"]
    require(a and len(a) == len(b), "unmatched workload length")
    failures = []
    max_error = 0.0
    for i, (left, right) in enumerate(zip(a, b)):
        require(len(left["rows"]) == len(right["rows"]), "unmatched batch sizes")
        for j, (expected, actual) in enumerate(zip(left["rows"], right["rows"])):
            require(expected["prompt"] == actual["prompt"], "unmatched input tokens")
            error = None
            if expected["tokens"] == actual["tokens"]:
                error = max(
                    abs(x - y)
                    for x, y in zip(
                        expected["logprobs"], actual["logprobs"], strict=True
                    )
                )
                max_error = max(max_error, error)
            if error is None or error > args.atol:
                failures.append({"batch": i, "request": j, "max_logprob_error": error})
    left = summarize(args.baseline, baseline)
    right = summarize(args.trial, trial)
    require(
        [r["device_total_lcm_blocks"] for r in left["rank"].values()]
        == [r["device_total_lcm_blocks"] for r in right["rank"].values()],
        "effective Device cache budgets differ",
    )
    for rank, data in right["rank"].items():
        for kind in args.require_copy:
            require(
                data["copies"].get(kind, {}).get("bytes", 0) > 0,
                f"rank {rank} has no completed {kind} payload",
            )
        if args.require_eviction:
            require(
                data["host_evictions"] > 0 and data["host_misses"] > 0,
                f"rank {rank} has no Host eviction/miss evidence",
            )
        if trial["config"]["debug_force_retraction_interval"]:
            pool = data["pool"]
            require(
                data["final_snapshot_free_lcm_blocks"]
                == pool["snapshot_bytes"] // pool["lcm_block_bytes"],
                f"rank {rank} retains snapshot blocks after the workload",
            )
    save(
        args.output,
        {
            "passed": not failures,
            "atol": args.atol,
            "failures": failures,
            "max_error_on_matching_sequences": max_error,
            "baseline": left,
            "trial": right,
        },
    )
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trial = sub.add_parser("run")
    trial.add_argument("--tp", type=int, choices=(1, 2), required=True)
    trial.add_argument("--physical-devices", type=hw2.parse_devices, required=True)
    trial.add_argument("--model", type=Path, required=True)
    trial.add_argument("--dtype", choices=("bfloat16", "float16"), required=True)
    trial.add_argument("--lm-head-compute-dtype", choices=("float32",))
    trial.add_argument("--memory-fraction", type=float, required=True)
    trial.add_argument("--device-tokens", type=int, required=True)
    trial.add_argument("--host-ratio", type=float, required=True)
    trial.add_argument("--snapshot-gb", type=float, default=0.0)
    trial.add_argument("--force-interval", type=int, default=0)
    trial.add_argument("--pattern", choices=("high", "low", "retract"), required=True)
    trial.add_argument("--rounds", type=int, default=12)
    trial.add_argument("--prompt-length", type=int, default=192)
    trial.add_argument("--output-tokens", type=int, default=16)
    trial.add_argument("--port", type=int, required=True)
    trial.add_argument("--output-dir", type=Path, required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("trial", type=Path)
    diff.add_argument("--output", type=Path, required=True)
    diff.add_argument("--atol", type=float, default=0.15)
    diff.add_argument(
        "--require-copy",
        action="append",
        default=[],
        choices=("WriteBackDone", "LoadBackDone", "SnapshotDone", "RestoreDone"),
    )
    diff.add_argument("--require-eviction", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return 0
    require(math.isfinite(args.atol) and args.atol >= 0, "invalid tolerance")
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
