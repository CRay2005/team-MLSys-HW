# Gemini Neon HW3 review — 2026-11-24

Requested by Steven. Reviewer: `gemini-4-neon-high` / Gemini 4 Neon (High), via
his configured Antigravity CLI. Two scoped read-only requests returned successfully.
The review covered the assignment and runner, then checker and tests, at source
state committed as `ab43a85`. No hardware results were supplied or certified.
An earlier combined request timed out without a review; a short connection check
then succeeded. Only final reviewer prose is preserved below, not internal agent
traces or tool logs. The follow-up fixes were validated locally by Codex; Gemini
has not separately signed off on the post-fix source.

## Disposition

| Finding | Decision and evidence |
| --- | --- |
| Two-prefix/high and never-revisit/low workloads lack a deliberate displacement/revisit sequence | Accepted as a coverage gap. Added explicit working set, pair rotation, bounded extension on later visits, and matched request counts/length profiles. This does not itself prove Host pressure or eviction. |
| Plan-only mode accepts static invalid devices/pairs | Fixed: validate the recorded 0..7 topology before output creation. Live occupancy remains a separate run-time check; recorded topology is not a discovery mechanism. |
| Candidate Host ratios bypass comparator validation; bool zero accepted in off reference | Fixed: require finite positive non-bool on ratios, complete numeric zero off settings, and an integer force interval. |
| Equal incomplete provenance can pass | Fixed: require complete harness/source/runtime/device schemas, SHA256-formatted model hashes, nonempty per-file source manifests and their recomputed aggregate digests. Optional Triton may explicitly be unavailable; torch/torch_npu may not. |
| Missing failure-path tests | Added: malformed modes, missing/mismatched provenance, malformed output counts/types, topology, pool settings, workload bounds and extension coverage. Full CPU suite: 32 passed. |
| Negative forced-retraction interval is invalid (both reviews) | Rejected. The pinned runtime explicitly uses N > 0 for decoding and N < 0 for prefill. Preserve -2 and add a positive test; reject bool/non-integer instead. |
| Suggested source_fingerprints values as nonempty strings | Adjusted to actual schema: HW2 records nested root/files/sha256 objects, not strings. Validation follows the producer's canonical file-map hash. |

Source for the signed interval:
[ServerArgs](../../homework-1/tokenspeed/python/tokenspeed/runtime/utils/server_args.py)
`--debug-force-retraction-interval` help says positive values retract a decoding
request and negative values retract a prefilling request between chunks.

No new transport implementation, runtime metric hooks, NPU offload results,
eviction proof, TP2 restore agreement or leak-free claim is established by this
review. Next work remains instrumentation and an early hardware smoke test.

## Reviewer output: runner / assignment

Three concrete defects/gaps found in the supplied HW3 harness:

### 1. Workload structurally cannot test idle-prefix L2 offload/restore, Host-eviction miss recovery, Host-budget sweeps, or post-restore continuation (`make_workload`)
- **Defect:** Even beyond the documented caveat that 2 prefixes may fit in HBM, `pattern="high"` runs the only 2 sessions (`[0, 1]`) simultaneously in every batch, while `pattern="low"` never revisits any prefix. As a result:
  1. No idle session is ever displaced by additional sessions to exercise normal HBM→Host prefix offload, reclamation, and on-demand restore (only forced retraction can evict).
  2. Neither mode can verify HW3’s required Host-eviction→miss-recovery on a revisited session or show capacity sensitivity when varying the Host budget (`high` has only 2 prefixes; `low` has 0 revisits).
  3. Revisits resend the exact 256-token prompt (100% match) rather than extending a cached prefix with new suffix tokens, missing hybrid KV/recurrent/conv state continuation past a restore boundary.
- **Smallest fix:** Add a `working_set` parameter (e.g., `>= 4`, defaulting to `6` or `8` for `high`) so consecutive batches cycle non-overlapping pairs across `range(working_set)` and revisit them in subsequent repeats; optionally append a short revisit suffix (keeping `len(prompt) + 16 <= 512`) to exercise partial-prefix restore + continuation.
- **Behavioral regression test:** Call `make_workload(seed, vocab_size=1024, pattern="high", repeats=2, working_set=6)` and assert: `> 2` unique sessions exist, consecutive batches do not pin all sessions simultaneously, every session is revisited in repeat 2 with a shared prefix + extended suffix (`+ 16 <= 512`), and `pattern="low"` remains all-distinct.

### 2. `--action plan` skips static device/TP2-topology validation (`main`)
- **Defect:** The static checks (`d not in range(8)` and `args.tp == 2 and devices[0] // 2 != devices[1] // 2`) execute only after `if args.action == "plan": return`. Consequently, `--action plan` creates `output_dir` and writes `status="planned_not_run"` for unrunnable specs (e.g., cross-card `--tp 2 --physical-devices 0,2` or out-of-range device `8`), deferring failure until `--action run`.
- **Smallest fix:** Extract static device-range and TP2 same-card validation into a helper invoked before `args.output_dir.mkdir(...)`, keeping only the live `npu-before.txt` occupancy check in the `run` path.
- **Behavioral regression test:** Invoke `main` with `--action plan --tp 2 --physical-devices 0,2` and `--action plan --tp 1 --physical-devices 8`; assert both raise `ValueError`/`RuntimeError` and leave `output_dir` uncreated.

### 3. Negative `force_interval` is accepted when `offloading="on"` (`build_config`)
- **Defect:** `build_config` validates `type(force_interval) is not int` (and forbids nonzero values for `offloading="off"`), but omits `force_interval < 0`. Passing `--offloading on --force-interval -1` passes validation and forwards `debug_force_retraction_interval=-1` into `Engine(**config)`.
- **Smallest fix:** Change the check to `if type(force_interval) is not int or force_interval < 0:`.
- **Behavioral regression test:** Assert `build_config(..., offloading="on", force_interval=-1, ...)` raises `ValueError`, while `force_interval=0` and `force_interval=4` succeed.

## Reviewer output: checker / tests

### 1. `compare_runs` does not validate `candidate` offloading settings (and allows `False == 0` in `reference`)
- **Defect**: `kvstore_ratio`, `retraction_snapshot_ratio`, and `debug_force_retraction_interval` are excluded from config equality via `allowed`, and `compare_runs` only checks `candidate["config"]["disable_kvstore"] is False`. A candidate JSON with `0`, negative, `NaN`/`inf`, `bool`, or deleted ratios (effectively disabling both Host pools while marked `"on"`), or an invalid `debug_force_retraction_interval`, silently passes. On `reference`, `reference["config"][key] != 0` also accepts `False` (`False == 0` in Python) and raises `KeyError` instead of `ValueError` on missing keys.
- **Smallest fix**: In `compare_runs`, verify each reference disable-key is present, `type(v) is not bool`, and `v == 0`; verify `candidate["config"]` has finite positive non-bool `int`/`float` values for both `kvstore_ratio` and `retraction_snapshot_ratio`, and a non-bool `int >= 0` for `debug_force_retraction_interval`.
- **Regression test**: Mutate `candidate["config"]` in `result_pair()` with `kvstore_ratio` or `retraction_snapshot_ratio` set to `0`, `-1`, `float("nan")`, `float("inf")`, `True`, or removed, and `debug_force_retraction_interval` set to `-1`/`True`; also set `reference["config"]["kvstore_ratio"] = False`. Assert `compare_runs(a, b)` raises `ValueError` for every case.

### 2. `compare_runs` checks provenance equality but not completeness or non-emptiness
- **Defect**: For `model_config_sha256`, `model_manifest_sha256`, `source_fingerprints`, `harness_fingerprints`, `runtime_versions`, and `physical_devices`, `compare_runs` only checks `reference[key] != candidate[key]`. If *both* results carry empty, null, or partial provenance—e.g. `source_fingerprints={}` (missing the three expected source trees), `harness_fingerprints={}`, `runtime_versions` missing/null for `torch`/`torch_npu`/`triton`, empty `model_*_sha256`, or `physical_devices=[]`—the check passes. (`result_pair()` itself currently uses incomplete stubs `{"python": "hash"}` and `{"torch": "test"}`.)
- **Smallest fix**: Validate each result before equality comparison: `model_config_sha256` and `model_manifest_sha256` are non-empty `str`; `harness_fingerprints` matches `set(harness_fingerprints())` with non-empty `str` values; `source_fingerprints` contains the three expected source-tree keys with non-empty `str` values; `runtime_versions` contains `{"torch", "torch_npu", "triton"}` with non-empty `str` values; `physical_devices` is a non-empty `list` of distinct non-negative `int`s (`type(d) is int`).
- **Regression test**: Update `result_pair()` to populate complete provenance, then symmetrically corrupt *both* `a` and `b` (empty/missing keys or `None`/`""` values in each provenance field; empty/duplicate/`bool` `physical_devices`) and assert `compare_runs(a, b)` raises `ValueError`.

### 3. Missing tests for existing mode, provenance-mismatch, and output-validation guards
- **Gap**: `OffloadTests` never exercises `compare_runs` rejecting `off`/`off`, `on`/`on`, or reversed `on`/`off` pairs; mismatches in `source_fingerprints`, `runtime_versions`, `model_config_sha256`, `physical_devices`, or tampered `workload_sha256`; or `validate_outputs` rejecting wrong `prompt_tokens`, negative/`bool` `output_ids`, negative/`bool` `cached_tokens`, and missing/extra batch outputs.
- **Smallest fix / regression test**: Add focused table-driven tests covering reversed/same-mode pairs, single-side mutations of every provenance field plus `workload_sha256`, and each untested `validate_outputs` failure mode, asserting `ValueError`.
