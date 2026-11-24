"""Recheck a completed 18-run matrix and summarize separately timed phases."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_offloading import compare_runs
from summarize_trace import summarize, validate_rank_agreement


def phases(result):
    groups = defaultdict(list)
    for index, (batch, row) in enumerate(zip(result["workload"]["batches"], result["runs"])):
        seconds = row["seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid batch duration")
        lengths = {len(ids) for ids in batch["input_ids"]}
        if len(lengths) != 1:
            raise ValueError("mixed prompt lengths in one batch")
        phase = "cold" if index == 0 else "initial" if batch["repeat"] == 0 else "revisit"
        key = f"{phase}_{lengths.pop()}"
        tokens = sum(len(output["output_ids"]) for output in row["outputs"])
        groups[key].append((seconds, tokens))
    return {key: dict(batches=len(rows), seconds=sum(t for t, _ in rows),
                      median_batch_seconds=statistics.median(t for t, _ in rows),
                      output_tokens_per_second=sum(n for _, n in rows) / sum(t for t, _ in rows))
            for key, rows in groups.items()}


def trace_report(directory, result):
    saved = json.loads((directory / "trace-summary.json").read_text())
    reports = []
    hashes = {}
    for file in sorted((directory / "trace").glob("*.jsonl")):
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        reports.append(json.loads(json.dumps(summarize([json.loads(line) for line in file.read_text().splitlines()]))))
    if hashes != saved["trace_sha256"] or reports != saved["ranks"]:
        raise ValueError("raw trace differs from recorded summary")
    validate_rank_agreement(reports, result["config"]["world_size"])
    if not saved["rank_completion_agreement"]:
        raise ValueError("rank completion disagreement")
    ops = Counter()
    for rank in reports:
        ops.update(rank["completed_operations"])
    return dict(
        device_blocks_per_rank=[r["final_scheduler"]["device_total"] for r in reports],
        allocation_per_rank=[r["allocation"] for r in reports],
        final_scheduler_per_rank=[r["final_scheduler"] for r in reports],
        d2h_bytes=sum(r["d2h_bytes"] for r in reports), h2d_bytes=sum(r["h2d_bytes"] for r in reports),
        completed_operations=dict(ops),
        first_admission_prefix_hits=sum(r["first_admission_prefix_hits"] for r in reports),
        first_admission_prefix_misses=sum(r["first_admission_prefix_misses"] for r in reports),
        observed_host_key_replacements=sum(r["observed_host_key_replacements"] for r in reports),
        replay_tokens=sum(r["replay_tokens"] for r in reports))


def analyze(root):
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "suite_passed" or status.get("suite") != "matrix" or status.get("jobs_completed") != 18:
        raise ValueError("matrix did not finish successfully")
    plan = json.loads((root / "results/experiment-plan.json").read_text())
    expected = {f"{p}-r{r}-{c}" for p in ("high", "low") for r in range(3)
                for c in ("off", "on-small", "on-large")}
    if len(plan) != 18 or {j["name"] for j in plan} != expected:
        raise ValueError("incomplete matrix plan")
    runs, metrics = {}, {}
    for name in sorted(expected):
        directory = root / "results" / name
        run = json.loads((directory / "result.json").read_text())
        if run["config"]["debug_force_retraction_interval"] != 0 or not run.get("observer", {}).get("enabled"):
            raise ValueError("performance requires normal scheduling and matched observation")
        if run["config"]["kvstore_ratio"] != (0 if name.endswith("-off") else 0.5 if name.endswith("-small") else 1):
            raise ValueError("unexpected Host budget")
        runs[name] = run
        metrics[name] = dict(phases=phases(run), **trace_report(directory, run))
    comparisons = {}
    for pattern in ("high", "low"):
        for repetition in range(3):
            baseline = f"{pattern}-r{repetition}-off"
            for mode in ("on-small", "on-large"):
                name = f"{pattern}-r{repetition}-{mode}"
                comparisons[name] = compare_runs(runs[baseline], runs[name])
                if metrics[name]["device_blocks_per_rank"] != metrics[baseline]["device_blocks_per_rank"]:
                    raise ValueError("unequal actual device block capacity")
    aggregate = {}
    for pattern in ("high", "low"):
        for mode in ("off", "on-small", "on-large"):
            keys = [f"{pattern}-r{r}-{mode}" for r in range(3)]
            baseline = [f"{pattern}-r{r}-off" for r in range(3)]
            aggregate[f"{pattern}-{mode}"] = {}
            phase_keys = metrics[keys[0]]["phases"].keys()
            if any(metrics[key]["phases"].keys() != phase_keys for key in keys + baseline):
                raise ValueError("unmatched phase lengths")
            for phase in phase_keys:
                values = [metrics[key]["phases"][phase]["output_tokens_per_second"] for key in keys]
                base_values = [metrics[key]["phases"][phase]["output_tokens_per_second"] for key in baseline]
                aggregate[f"{pattern}-{mode}"][phase] = dict(
                    independent_runs=3, tokens_per_second=values,
                    median_tokens_per_second=statistics.median(values),
                    min_tokens_per_second=min(values), max_tokens_per_second=max(values),
                    median_paired_throughput_ratio_to_off=statistics.median(a / b for a, b in zip(values, base_values)))
    return dict(runs=metrics, comparisons=comparisons, aggregate=aggregate,
                limits=["Instrumented synthetic greedy-token workloads; timing includes observation and replay.",
                        "Cold first batch, remaining initial prompts and extended revisits remain separate.",
                        "Three sequential fresh Engine repetitions; order is not randomized and shared-server contention may drift.",
                        "Prefix hits span all cache tiers; physical Host slot replacements are not all logical eviction events.",
                        "Logical Host tensor allocations do not measure physical pinned reserved memory or RSS.",
                        "A workload described as high reuse may still miss if both cache budgets are too small."],
                full_assignment_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.suite_dir)
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(runs=len(result["runs"]), comparisons=len(result["comparisons"]))))


if __name__ == "__main__":
    main()
