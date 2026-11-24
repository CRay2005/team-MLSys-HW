# Homework 3 — HBM/Host cache offloading

Status, 2026-11-24: assignment evaluated; initial experiment runner and 57 local
tests implemented. **0.8B and 4B TP1 baseline/forced restoration passed on Ascend; the 18-run
0.8B performance matrix passed with zero Host prefix hits. TP2 and complete
offload lifecycle validation remain pending.**
Due 2026-11-26. Original requirements: [assignment.md](assignment.md).

[Contribution statement and report draft](report.md) credits Steven's new HW3
tooling and experiments and distinguishes the reused upstream/group runtime.

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
- High reuse cycles pairs through an explicit working set, rotating session order
  on repeat visits and optionally extending their prefixes. Low reuse uses fresh
  session prefixes throughout, with matched request counts and length changes.
  Both use synthetic valid-token inputs from the HW2 seed, not a language-quality
  benchmark. Use at least four sessions for displacement experiments; two is
  supported only as a small smoke configuration.
- Records raw outputs, batch latency, configuration/source hashes, device/process
  snapshots, and an explicit list of missing lifecycle metrics.
- `compare_results.py` checks every generated token under matched configurations.
  It never treats cached-token metadata or token agreement as proof of a Host hit.

The checker also rejects empty/duplicate workloads, mismatched harness/helper
versions, incomplete provenance, invalid on-mode budgets and an off baseline with
an absolute snapshot allocation. Hardware runs
must import the exact source roots fingerprinted in the result. Each save replaces
a complete JSON file, preserving the previous record if serialization fails.
Five tests exercise the actual upstream Host-pool sizing implementation without
torch or accelerator mocks; the remaining tests cover the harness and its CLI.

Gemini 4 Neon independently reviewed the runner and checker. See the
[findings and dispositions](docs/gemini-neon-review-20261124.md); recommendations
were checked against source, including rejection of an incorrect negative-interval
restriction. The review is not hardware certification.

This milestone adds a harness, not a new transport implementation. Initial NPU
generation and forced-suspension token comparisons now pass. Completed-copy measurements and final quiescence now pass in an independent
instrumented TP1 smoke; eviction, long-cycle leak freedom and TP2 remain pending.
Even the larger working set may fit entirely
in HBM; calibrate pressure before claiming a Host-cache benefit.

## First Ascend results — November 24

[Verified results and limitations](results/2026-11-24-tp1-smoke/summary.json):
0.8B BF16 with FP32 recurrent state, TP1 on physical chip 1. Baseline, forced
decode (`2`) and forced prefill (`-2`) each completed eight requests with 16 output
tokens. Both candidates matched every baseline token: 16 request comparisons,
256 output-token comparisons. Comparisons passed again on the downloaded files.

All modes allocated the same 213,909,504-byte device arena with 16 LCM blocks.
The decode log records 179 snapshot-store/restore starts; the prefill log records
four. These are **started-operation logs**, not completed-transfer byte/timing
measurements. The checker therefore retains `offload_lifecycle_verified: false`.
These forced runs are correctness probes, not performance comparisons.

All 49 downloaded evidence files passed checksum verification, including three
failed setup/calibration trials. The final NPU snapshot confirms chip 1 was free
again. Source/model checks and 32 core CPU tests also passed on the server.
At that initial milestone, TP2, larger models, Host eviction/misses, repeated-cycle
checks and the performance matrix were outstanding. Later sections record the
subsequent 4B and matrix results; TP2 and full lifecycle coverage remain open.

Gemini reviewed the original harness earlier. The new launcher review attempt
failed when the CLI rejected the previously working Neon model identifier and
model discovery timed out; no new Gemini signoff is claimed.

## Local checks

### Run the first server experiments independently of chat

After connecting iSecSP, run from this Mac's Terminal:

```bash
cd ~/Projects/MLSys-HW
python3 homework-3/scripts/launch_smoke.py --host YOUR_COURSE_USER@YOUR_COURSE_HOST
```

Uses the existing SSH key and known-host entry on port 2222. The default physical
chip is 1; `--device N` selects another previously verified course chip. It stops
if the chip is occupied or occupancy cannot be parsed. This is not a reservation.
The launcher uploads a small checksummed harness into a unique server directory,
copies the existing HW2 runtime into that directory and verifies all three Python
source trees against local hashes. Existing HW1/HW2 runs are preserved. It reuses
the server's private environment, CANN and 0.8B model; missing or different files
stop the suite rather than trigger installation or download.

The detached suite runs 40 core/observer tests, then three 0.8B TP1 cases:
off, forced decode suspension (`2`), and forced prefill suspension (`-2`), checking
all generated tokens against off. Each model invocation has a ten-minute limit;
failures stop the sequence. Four sessions and two cycles provide an initial smoke,
not a calibrated memory-pressure benchmark. All three runs use the HW2-tested
16,384-token cap: the initial 1,024-token trial produced only two hybrid-cache
pages and could not admit even one request. Only this launcher's child process
groups are cleaned up. Available engine/HCCL ports are selected and rechecked
before each run. A per-account lock prevents overlapping smoke launchers.

Wait for the printed **Started independently** confirmation before closing the
Terminal. Once started, server jobs survive VPN/chat loss. The command prints
exact status and result-download commands, and keeps a local launch receipt in
`.local-review/hw3-smoke-*`. If connection is lost during launch, inspect the
printed server directory before retrying: launch may have succeeded without its
acknowledgment reaching the Mac. `suite.log`, `status.json`, and `results/` remain
on the server. Missing status can mean the environment setup failed; read the log.

`--prepare-only` builds the package without contacting the server. The eleven
local launcher tests cover suite plans, bounded queue transitions, packaging, occupied/unknown device refusal, failure
propagation, port conflict handling and real subprocess timeout cleanup. Initial
server preflight passed source/model verification and all 32 harness tests, then
stopped on a port conflict before NPU use; automatic port selection was added.
A successful smoke does **not** certify actual offload completion,
eviction, leak freedom, TP2, larger models or performance; those experiments remain.

`--suite trace` repeats the smoke with optional completed-transfer and scheduler
observation. See [measurement definitions and limitations](docs/runtime-observer.md).
Eight additional CPU tests validate completion accounting and reject incomplete
transfers or restoration-frontier mismatches, including disagreement between ranks. The observer passed an independent Ascend TP1 smoke with 165 decode and 7 prefill
snapshot/restore frontier checks, completed payload bytes and fresh final idle
snapshots. All outputs match the baseline; all device blocks are reclaimable and
all snapshot blocks free at the end. See [trace evidence](results/2026-11-24-tp1-trace/summary.json).
Earlier smoke evidence remains separately preserved. Full assignment coverage is
still incomplete; physical replacement counts are not complete logical evictions.

Additional suites use the same verified deployment, occupancy checks and bounded
process supervision:

- `--suite matrix`: 18 fresh Engine runs: high/low reuse, off/on L2 ratio 0.5/on
  L2 ratio 1, three independent repetitions. Snapshot ratio stays 1 for both on
  cases. Eight sessions, two cycles, force interval 0; timing includes observer
  overhead and separates 256-token and 264-token batches during analysis.
- `--suite stress`: eight sessions over eight cycles, off/decode/prefill probes,
  with fresh-quiescence and exact-output checks after each run.
- `--model-size 4B`: requires the pinned model already downloaded and verified in
  the server workspace; uses FP16 weights/KV and FP32 recurrent state/LM head.
- `--tp 2 --device 0` (or first chip 2/4/6): checks both chips of the recorded
  same-card pair. It stops if either is occupied; it does not wait or evict users.

These are runnable experiment plans, not evidence until their results pass.

From the repository root:

```bash
.venv/bin/python -m unittest discover -s homework-3/tests -v
.venv/bin/python homework-3/scripts/run_offloading.py \
  --action plan --model /path/to/Qwen3.5-0.8B \
  --model-config homework-2/metadata/Qwen3.5-0.8B/config.json \
  --seed-cases homework-2/cases.json --tp 1 --physical-devices 1 \
  --memory-fraction 0.12 --device-tokens 1024 --port 31893 \
  --offloading on --host-ratio 1 --snapshot-ratio 1 --force-interval 2 \
  --dtype bfloat16 --head-dtype native --pattern high --repeats 2 \
  --working-set 8 --revisit-suffix-tokens 8 \
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
steady-state performance and run at least three fresh repetitions. All first-cycle
sessions are new, and extension changes prompt length: report 256-token initial
and 264-token extended requests separately for the example above. Do not pool
them into one latency comparison. Identical schedule/lengths must be used for off
and on. `--repeats` cycles within one Engine instance; independent repetitions
require separate fresh invocations and output directories.

Analyze a completed downloaded matrix with:

```bash
.venv/bin/python homework-3/scripts/analyze_matrix.py \
  --suite-dir /path/to/downloaded-suite --output /path/to/new-analysis.json
```

The analyzer independently recomputes raw trace summaries, validates exact output
comparisons and actual device block budgets, and keeps cold, initial and extended
phases separate. Throughput is generated tokens divided by total batch time within
each phase. It reports three independent run values and paired ratios, discloses
fixed execution order and observer overhead, and rejects incomplete suites.
Four CPU tests cover aggregation and incomplete/invalid timing input.

For checksum-verified evidence retrieval after a suite finishes or fails:

```bash
.venv/bin/python homework-3/scripts/collect_evidence.py \
  --host YOUR_COURSE_USER@YOUR_COURSE_HOST --run PRINTED_RUN_LABEL \
  --output .local-review/NEW_EVIDENCE_DIRECTORY
```

This refuses running suites, streams result/log evidence without making a server
archive, verifies every file against its manifest before extraction, and removes
the temporary local archive after verification. It does not download model weights
or the copied runtime. Two tests cover valid/tampered payloads and unsafe paths.

`--after-run PRINTED_RUN_LABEL` queues a suite behind a named successful suite.
The detached controller waits at most 30 minutes without holding the per-account
lock. A failed/blocked predecessor cancels the queued suite. Source/model hashes,
ports and free-chip occupancy are still checked before actual execution; queuing
does not reserve a chip. The queue transition test covers success, failure,
timeout and invalid predecessor paths.

November 24 matrix result: all 18 normal-scheduling TP1 runs and 12 output pairs
passed. All modes had the same 204 MiB device arena and 16 usable LCM blocks.
The 8- and 16-block L2 budgets produced writebacks/churn but **zero Host loadbacks**
and zero prefix hits in both patterns. This is a valid negative result, not
an offloading speedup. Cold/initial/extended timings remain separate; see the
[matrix analysis](results/2026-11-24-tp1-matrix/analysis.json) and [report](report.md).

`--suite capacity` is a short follow-up to the zero-hit matrix: three high-reuse
runs with off/L2 ratio1/L2 ratio8, eight sessions and two cycles, normal scheduling
and the same device budget. Snapshot ratio remains1. This tests whether a larger
Host retention budget enables actual loadbacks; one run per budget is exploratory,
not a three-repetition performance result. Its maximum logical L2 pool is about
1.5 GiB for the 0.8B geometry, plus the separate192 MiB snapshot pool.

The 4B TP1 trace suite also passed off/decode/prefill exact-output comparisons,
149 decode plus5 prefill restoration-frontier checks, transfer completion and
final quiescence. All modes use the same544 MiB device arena/16LCM blocks.
See [4B evidence](results/2026-11-24-4B-tp1-trace/summary.json). These forced runs
are correctness evidence, not performance results; TP2 remains outstanding.
