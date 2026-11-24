# Homework 3 — HBM/Host cache offloading

Status, 2026-11-24: assignment evaluated; initial experiment runner and 23 local
tests implemented. **No HW3 Ascend execution or end-to-end offload result yet.**
Due 2026-11-26. Original requirements: [assignment.md](assignment.md).

## Assessment

This requires real offload → device-block reuse → restore, correct hybrid state,
bounded Host storage, misses/eviction, TP2, and comparison on 0.8B plus at least
one larger model. Unlike HW2, it does not demand all four model sizes.

The pinned source already contains Host L2, a separate retraction snapshot pool,
NPU DMA selection, mixed-state cache recipes and replica completion agreement.
Reuse and validate these facilities before adding another cache manager. Their
presence does not prove they work on this server. HW2 disabled prefix caching
and KVStore, so its results do not validate this path.

The main feasibility risk is the two days remaining for hardware debugging.
Start with 0.8B TP1, then 0.8B TP2 and 4B TP2. Reserve a free same-card pair early.
Use 4B's HW2-tested FP16 weights/KV, FP32 recurrent state and FP32 final projection.
See [source audit and execution plan](docs/implementation-plan.md).

## First implementation

`scripts/run_offloading.py` uses the existing Engine and HW2 provenance helpers.
Plan-only operation needs no torch; hardware execution is explicitly selected.

- Off disables **both** KVStore and retraction snapshots. Device prefix caching
  stays enabled in both modes, isolating the effect of Host storage.
- On uses the existing direct/DMA transport and explicit L2/snapshot ratios.
  Device token and memory budgets stay fixed across each comparison.
- High reuse revisits two distinct prefixes and reverses batch order. Low reuse
  uses eight unique prefixes per repeat. Both use synthetic valid-token inputs
  derived from the saved HW2 seed, not a language-quality benchmark.
- Records raw outputs, batch latency, configuration/source hashes, device/process
  snapshots, and an explicit list of missing lifecycle metrics.
- `compare_results.py` checks every generated token under matched configurations.
  It never treats cached-token metadata or token agreement as proof of a Host hit.

The checker also rejects empty/duplicate workloads, mismatched harness/helper
versions and an off baseline with an absolute snapshot allocation. Hardware runs
must import the exact source roots fingerprinted in the result. Each save replaces
a complete JSON file, preserving the previous record if serialization fails.
Five tests exercise the actual upstream Host-pool sizing implementation without
torch or accelerator mocks; the remaining tests cover the harness and its CLI.

This first milestone adds a harness, not a new transport implementation. NPU
Engine execution, eviction, real suspension/resumption, leak freedom and full
numerical validation remain pending. The two-prefix workload may fit entirely in
HBM; calibrate pressure before claiming a Host-cache benefit.

## Local checks

From the repository root:

```bash
.venv/bin/python -m unittest discover -s homework-3/tests -v
.venv/bin/python homework-3/scripts/run_offloading.py \
  --action plan --model /path/to/Qwen3.5-0.8B \
  --model-config homework-2/metadata/Qwen3.5-0.8B/config.json \
  --seed-cases homework-2/cases.json --tp 1 --physical-devices 1 \
  --memory-fraction 0.12 --device-tokens 1024 --port 31893 \
  --offloading on --host-ratio 1 --snapshot-ratio 1 --force-interval 2 \
  --dtype bfloat16 --head-dtype native --pattern high --repeats 1 \
  --output-dir .local-review/hw3-first-plan
```

`plan` writes only configuration/workload. Model paths and devices are placeholders
until checked on the server. Every invocation requires a fresh output directory.

For hardware runs, use the existing course Python/CANN environment and the three
source PYTHONPATH roots documented in HW2. Verify deployment hashes and actual
topology, set a distinct HCCL socket range, capture full stdout/stderr and use an
external bounded timeout. Replace `--action plan` with `--action run` and use real
model/config paths and a new results directory. The runner checks current NPU
occupancy and refuses occupied devices or an invalid pair under the previously
recorded four-card/two-chip mapping. This is not a device reservation; coordinate
shared-server use and reverify the mapping before using another server.

Create the reference with `--offloading off --host-ratio 0 --snapshot-ratio 0
--force-interval 0`. Compare it with the otherwise matching on run. Use intervals
`2` and `-2` only for decode/prefill correctness probes; an interval does not prove
a retraction occurred. Require completed snapshot/restore events. Use interval
`0` for normal performance tests.

```bash
.venv/bin/python homework-3/scripts/compare_results.py \
  --reference /path/to/off/result.json --candidate /path/to/on/result.json \
  --output /path/to/new-comparison.json
```

`max_total_tokens` is a requested cap. Record actual allocated block budgets on
each rank before claiming equal HBM capacity. Batch timings include replay but
do not isolate restore latency. The first batch is cold; keep it separate from
steady-state performance and run at least three fresh repetitions.
