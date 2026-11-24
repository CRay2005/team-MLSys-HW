"""Summarize observed completions without equating submissions with transfers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(events):
    if not events or events[0]["event"] != "observer_start":
        raise ValueError("missing observer start")
    begins, completed = {}, set()
    byte_counts, durations, operations = Counter(), Counter(), Counter()
    final_stats = None
    peaks = Counter()
    allocation = None
    first_requests = {}
    frontiers, restore_checks = {}, []
    host_entries, device_owners = {}, {}
    host_replacements = 0
    device_changes = 0
    replay_tokens = 0
    request_ranges = []
    agreement = []
    for event in events:
        kind = event["event"]
        if kind == "allocation":
            allocation = event
        elif kind == "transfer_begin":
            if event["batch"] in begins:
                raise ValueError("duplicate transfer batch")
            begins[event["batch"]] = event
        elif kind == "transfer_failed":
            raise ValueError("trace contains failed transfer")
        elif kind == "transfer_complete":
            batch = event["batch"]
            if batch not in begins or batch in completed:
                raise ValueError("unknown/duplicate completion")
            original = begins[batch]
            if any(event[key] != original[key] for key in ("kind", "op_ids", "payload_bytes")):
                raise ValueError("completion metadata differs from submission")
            completed.add(batch)
            byte_counts[event["kind"]] += event["payload_bytes"]
            durations[event["kind"]] += event["observed_seconds"]
            operations[event["kind"]] += len(event["op_ids"])
        elif kind in ("scheduler", "scheduler_before_close"):
            for key in ("device_active", "host_cached", "host_pinned", "retracted", "tail_cap_evictions"):
                peaks[key] = max(peaks[key], event[key])
            final_stats = event
        elif kind == "rank_agreed_completions":
            agreement.extend((e["kind"], e["op_id"]) for e in event["operations"])
        elif kind == "cache_plan":
            if event["kind"] == "SnapshotOp":
                for rid, frontier in zip(event["request_ids"], event["frontiers"]):
                    frontiers[rid] = frontier
            if event["kind"] == "RestoreOp":
                for rid, frontier in zip(event["request_ids"], event["frontiers"]):
                    if rid not in frontiers:
                        raise ValueError("restore without a recorded snapshot frontier")
                    restore_checks.append(dict(request_id=rid, stored=frontiers[rid], restored=frontier))
            if event["kind"] == "WriteBackOp":
                for groups, blocks, hashes, offsets in zip(event["group_ids"], event["destination_blocks"],
                                                          event["content_hashes"], event["page_offsets"]):
                    for group, block, content_hash, offset in zip(groups, blocks, hashes, offsets):
                        key, value = (group, block), (content_hash, offset)
                        if key in host_entries and host_entries[key] != value:
                            host_replacements += 1
                        host_entries[key] = value
        elif kind == "forward":
            request_ids = event["request_ids"]
            for i, rid in enumerate(request_ids):
                if i < event["extend_count"]:
                    start, length = event["prefix_lengths"][i], event["input_lengths"][i]
                    first_requests.setdefault(rid, start)
                    replay_tokens += event["replay_lengths"][i]
                    request_ranges.append(dict(request_id=rid, start=start, end=start + length,
                                               replay_length=event["replay_lengths"][i]))
                for group, tables in event["block_tables"].items():
                    for block in tables[i]:
                        if block == 0:
                            continue
                        key = (group, block)
                        if key in device_owners and device_owners[key] != rid:
                            device_changes += 1
                        device_owners[key] = rid
    if set(begins) != completed:
        raise ValueError(f"{len(set(begins) - completed)} transfers lack observed completion")
    if final_stats is None or final_stats["event"] != "scheduler_before_close":
        raise ValueError("missing final scheduler snapshot")
    if any(item["stored"] != item["restored"] for item in restore_checks):
        raise ValueError("snapshot/restore frontier mismatch")
    return dict(rank=events[0]["rank"], allocation=allocation, peaks=dict(peaks), final_scheduler=final_stats,
                completed_payload_bytes=dict(byte_counts), completed_operations=dict(operations),
                observed_completion_seconds_sum=dict(durations),
                timing_note="Call entry to observed ACK; includes queueing, polling and instrumentation. Overlaps must not be summed as wall time.",
                d2h_bytes=byte_counts["writeback"] + byte_counts["snapshot"],
                h2d_bytes=byte_counts["loadback"] + byte_counts["restore"],
                transfer_batches=len(completed), frontier_checks=restore_checks,
                first_admission_prefix_hits=sum(v > 0 for v in first_requests.values()),
                first_admission_prefix_misses=sum(v == 0 for v in first_requests.values()),
                prefix_hit_note="Any cache tier; loadback operation counts separately show Host prefix loads, not a per-request Host hit counter.",
                observed_host_key_replacements=host_replacements,
                replacement_note="Host destinations assigned different keys; counts slot replacements, not all logical eviction events.",
                observed_device_owner_changes=device_changes,
                owner_note="Block IDs assigned to different request IDs; shared prefixes can also cause owner changes.",
                replay_tokens=replay_tokens, prefill_ranges=request_ranges,
                rank_agreed_operations=sorted(agreement), complete_assignment_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.run_dir / "trace").glob("*.jsonl"))
    if not files:
        raise ValueError("no runtime traces; observation may not have loaded")
    reports = []
    hashes = {}
    for file in files:
        reports.append(summarize([json.loads(line) for line in file.read_text().splitlines()]))
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    result = json.loads((args.run_dir / "result.json").read_text())
    if len(reports) != result["config"]["world_size"] or len({r["rank"] for r in reports}) != len(reports):
        raise ValueError("wrong/missing trace rank count")
    output = dict(ranks=reports, trace_sha256=hashes,
                  rank_completion_agreement=all(r["rank_agreed_operations"] == reports[0]["rank_agreed_operations"] for r in reports),
                  d2h_bytes=sum(r["d2h_bytes"] for r in reports), h2d_bytes=sum(r["h2d_bytes"] for r in reports),
                  full_assignment_verified=False)
    (args.run_dir / "trace-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"ranks": len(reports), "d2h_bytes": output["d2h_bytes"], "h2d_bytes": output["h2d_bytes"]}))


if __name__ == "__main__":
    main()
