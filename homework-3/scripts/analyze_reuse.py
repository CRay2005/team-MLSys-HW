"""Find strict TP1 writeback/reuse/loadback witnesses in verified traces."""
import argparse
import json
from pathlib import Path

from analyze_matrix import trace_report


def witnesses(events, markers):
    """Require completed copies and a different-prefix owner between them."""
    owners, host, writes, loads = {}, {}, {}, {}
    groups, seen_requests, candidates = {}, set(), []
    for event in events:
        kind, now = event["event"], event["monotonic_ns"]
        if kind == "allocation":
            groups = {i: group["name"] for i, group in enumerate(event["groups"])}
        elif kind == "cache_plan" and event["kind"] in ("WriteBackOp", "LoadBackOp"):
            for op, gs, sources, destinations in zip(event["op_ids"], event["group_ids"],
                                                    event["source_blocks"], event["destination_blocks"]):
                for group, source, destination in zip(gs, sources, destinations):
                    name = groups[group]
                    if event["kind"] == "WriteBackOp":
                        row = dict(group=name, original_device_block=source, host_block=destination,
                                   original_request=owners.get((name, source)), writeback_op=op,
                                   writeback_ack_ns=None, reuse=None)
                        host[name, destination] = row
                        writes.setdefault(op, []).append(row)
                    else:
                        stored = host.get((name, source))
                        if stored and stored["writeback_ack_ns"] and stored["reuse"]:
                            row = dict(stored, loadback_op=op, restored_device_block=destination,
                                       loadback_plan_ns=now, loadback_ack_ns=None)
                            loads.setdefault(op, []).append(row)
        elif kind == "transfer_complete":
            mapping = writes if event["kind"] == "writeback" else loads if event["kind"] == "loadback" else {}
            field = "writeback_ack_ns" if event["kind"] == "writeback" else "loadback_ack_ns"
            for op in event["op_ids"]:
                for row in mapping.get(op, []):
                    row[field] = now
        elif kind == "forward":
            for i, rid in enumerate(event["request_ids"]):
                marker = markers[rid]
                blocks = {(group, block) for group, tables in event["block_tables"].items()
                          for block in tables[i] if block != 0}
                for stored in host.values():
                    original = stored["original_request"]
                    if (original in markers and markers[original] != marker
                            and stored["writeback_ack_ns"] is not None
                            and stored["writeback_ack_ns"] < now and stored["reuse"] is None
                            and (stored["group"], stored["original_device_block"]) in blocks):
                        stored["reuse"] = dict(request_id=rid, first_token=marker, forward_ns=now)
                if rid not in seen_requests and i < event["extend_count"] and event["prefix_lengths"][i] > 0:
                    for rows in loads.values():
                        for row in rows:
                            if (row["original_request"] in markers and markers[row["original_request"]] == marker
                                    and row["reuse"]["forward_ns"] < row["loadback_plan_ns"]
                                    and (row["group"], row["restored_device_block"]) in blocks):
                                candidates.append((row, rid, marker, event["prefix_lengths"][i], now))
                seen_requests.add(rid)
                for block in blocks:
                    owners[block] = rid
    # Forward records are CPU plans, emitted before the executor starts transfers.
    # Require the subsequent actual ACK, without pretending planning is kernel execution.
    return [dict(row, resumed_request=rid, resumed_first_token=marker,
                 resumed_prefix_tokens=prefix, resumed_forward_plan_ns=now)
            for row, rid, marker, prefix, now in candidates
            if row["loadback_ack_ns"] is not None and row["loadback_ack_ns"] > row["loadback_plan_ns"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run_dir / "result.json").read_text())
    if result["config"]["world_size"] != 1:
        raise ValueError("This witness analysis is scoped to TP1")
    trace_report(args.run_dir, result)
    markers = {output["meta_info"]["id"]: ids[0]
               for batch, run in zip(result["workload"]["batches"], result["runs"])
               for ids, output in zip(batch["input_ids"], run["outputs"])}
    files = list((args.run_dir / "trace").glob("*.jsonl"))
    rows = witnesses([json.loads(line) for line in files[0].read_text().splitlines()], markers)
    report = dict(witnesses=rows, block_witnesses=len(rows), resumed_requests=len({r["resumed_request"] for r in rows}),
                  scope="TP1: completed writeback -> different-first-token forward plan using old device block -> original-prefix loadback/continuation plan, with actual loadback ACK",
                  limitation="Forward records are CPU plans, not kernel timestamps; executor stream fences order actual copies and computation. Exact output correctness must be checked against off separately.")
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("block_witnesses", "resumed_requests")}))


if __name__ == "__main__":
    main()
