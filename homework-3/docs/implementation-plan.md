# HW3 source audit and evidence plan

Reviewed 2026-11-24. Outer baseline: `e228c25`; TokenSpeed upstream:
`52bdb9763bb2da3d61336398072671885cc0a33e`, with frozen HW1 and HW2 patches.
The existing MacBook checkout is used for bounded editing/CPU checks. No new
checkout, environment, model download or persistent service was created.

## Existing integration points

Paths relative to `homework-1/tokenspeed/`:

| Component | Source finding | HW3 action |
| --- | --- | --- |
| Host storage | `python/tokenspeed/runtime/cache/host_storage.py`, `cache/l2/sizing.py`: pinned packed L2 and snapshot allocation | Measure bytes including padding, null blocks and slot blobs |
| NPU transfer | `tokenspeed-kernel/python/tokenspeed_kernel/ops/kvcache/host_transfer.py`: `prepare_backend` selects DMA on NPU; `_transfer_dma` copies byte slices | Test pinned copies and event ordering on Ascend with `direct` transport |
| Lifecycle | `python/tokenspeed/runtime/cache/l2/executor.py`: writeback/loadback/snapshot/restore lanes and completion ACKs | Add completed-operation byte/timing measurements without changing fences |
| Hybrid state | `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py`: attention, SSM and conv fields in shared arena | Confirm every required field transfers at matching context position and dtype |
| Slot state | `python/tokenspeed/runtime/execution/slot_state.py` and executor blob arena | Include required non-cache state when restoring into a different slot |
| Rank agreement | `python/tokenspeed/runtime/engine/cache_hooks.py`: replica intersection of completed operations | Verify both ranks complete the same ticket/context before resume |
| Controls | `python/tokenspeed/runtime/utils/server_args.py`: L2 budget, separate snapshot budget and debug force interval | Off baseline must disable both Host pools; no L3 service needed |

Design references: `docs/design/cache-concepts.md` and `docs/design/scheduler.md`.
No new adaptation patch is claimed in this first milestone. Reproduce the existing
minimal NPU path, then fix demonstrated gaps in the existing framework.

## Mixed-state boundary and lifecycle

| State | Saved unit / position | Recovery |
| --- | --- | --- |
| Attention K/V | Physical group blocks with token coverage | Copy all referenced fields; restore block tables; KV alone cannot establish a complete hybrid prefix |
| GDN recurrent state | Working/published state slots at their scheduler frontiers | Exact suspension restores both required slots; prefix reuse uses a checkpoint common to every state group |
| Convolution history | Conv fields in the associated state groups | Restore the same context boundary and original activation dtype |
| Other slot state | Exporter-defined blob | Restore into the newly allocated slot for the same request |

For prefix reuse, safe position T is the intersection of KV coverage and state
checkpoints. Record requested/available/restored T and replay interval
[T, prompt end); measure replay separately and include it in end-to-end latency.
For exact suspended-request images, verify that the saved frontier is restored
without silently rewinding. TP2 ranks must agree on that frontier.

A Host prefix miss recomputes from the remaining safe prefix or zero. A suspended
request image is pinned, not an ordinary evictable prefix. If its snapshot pool
cannot reserve capacity, the existing scheduler can abort; retain this failure
and do not describe it as successful miss handling.

Quiescence and copy completion are distinct. The existing scheduler can reassign
blocks within a plan while stream fences protect the new owner's writes. Preserve
those fences: submitting a copy is not completion. Keep Host destinations and
restore sources owned until required transfers/rank acknowledgements complete.
Record block IDs and successive owners to prove reuse when preallocated HBM stays
constant in `npu-smi`.

## Required experiments — verified TP1 results and remaining gaps

November 24 update: the baseline and forced decode/prefill runs on 0.8B TP1
completed, with eight requests each and exact token agreement for both candidates.
The logs show suspension, writeback and restore starts. Both comparisons were
reverified after checksum-verified download; chip 1 was released. See
[the evidence summary](../results/2026-11-24-tp1-smoke/summary.json).
Follow-up instrumented 0.8B and 4B TP1 suites now verify completed payload bytes,
snapshot/restore frontiers and final quiescence. The18-run normal-scheduling
0.8B matrix also passed, but both Host budgets produced zero prefix hits and
zero loadbacks. See the current [report](../report.md) for measurements and scope.
The8-cycle stress and larger-Host retention probe now also pass. L2ratio8
produced8completed loadbacks and8prefix hits, with18 cross-session device-block
reuse witnesses across7resumed requests. TP2 and complete Host logical-eviction
accounting remain outstanding; CPU plan timestamps are not kernel timestamps. The initial 1,024-token
budget could not admit a request; the successful smoke used a fixed 16,384-token
cap in all three modes. Do not interpret this as calibrated Host-cache pressure.

| Requirement | Acceptance evidence |
| --- | --- |
| Same-session restore | 0.8B TP1 forced prefill/decode suspension, completed store/restore, matching outputs against off run |
| Isolation | Distinct prefixes, reversed batch order and forced suspension; per-request output and slot ownership checks |
| Eviction/miss | Exceed small Host budget, observe eviction, revisit evicted prefix, observe safe miss/replay and correct output |
| Repeated lifecycle | Warmed repeated offload/restore, bounded steady-state block/Host usage; distinguish retained cache from leaked references |
| TP2/larger model | 0.8B TP2 and 4B TP2, both-rank transfer/context evidence and matched outputs |
| Capacity/performance | Equal measured device budgets; high/low reuse; off versus on at two Host budgets; >=3 repetitions |

Use 0.8B BF16/FP32-state and 4B FP16/FP32-head from HW2. Initial trial Host ratios:
L2=1 versus 2, snapshot=1 held fixed. These do not guarantee pressure or fit: size
from real group byte geometry and available Host RAM. Enlarge the working set or
reduce Host capacity until eviction actually occurs. A two-prefix workload that
fits on device cannot establish a Host benefit.

After Gemini Neon review, the harness accepts an explicit even working set
(2..64; use >=4 for displacement) and a bounded revisit suffix. High-reuse cycles
now leave sessions idle between visits and extend their prompts after the first
cycle. Low reuse has matching request counts/lengths and fresh prefixes. Separate
initial versus extended requests in analysis and repeat whole runs independently.
This improves coverage; observed eviction/restore events are still required.

The 32 CPU checks now include actual upstream pool sizing, strict provenance and
Host-mode validation, early static topology rejection, and negative force intervals
for prefill. [Gemini review and disposition](gemini-neon-review-20261124.md) records
which suggestions were accepted and why the interval restriction was rejected.

The optional observer now records device total/active/available blocks, logical
Host pool bytes and cached/pinned entry counts, completed D2H/H2D payloads and
host-observed ACK timing, request/operation IDs and restore/prefill positions.
Prefix hits are across tiers; physical Host destination replacements are not a
complete logical-eviction counter. Private-block reuse causality and physical Host
RSS/reserved allocation remain unproven. The replay-length field denotes explicit
state replay, not every token recomputed after a prefix miss; a miss starting at
zero recomputes the prompt and its time is included in batch latency.
Aggregate bytes across ranks, but do not sum overlapping rank latencies as wall
time. Keep transfer duration separate from end-to-end restoration/admission.

If off fails at a large working set, retain the failure. Use a second workload
where both modes succeed for fair latency comparison. Separate cold and warm
measurements and exclude artificial forced retractions from normal performance.

## Next steps

1. TP1 smoke, instrumented restoration on0.8B/4B, the18-run normal-scheduling
   matrix,8-cycle stress and larger-Host capacity probe are verified and locally
   committed. All scheduled suites finished and physical chip1 was released.
2. Run TP2 when a same-card pair is free. The final occupancy check found only
   chip1 free, so none of0/1,2/3,4/5,6/7 was available. Validate both-rank copy
   completions, matching restore frontiers and exact output comparisons.
3. Close the full Host logical-eviction accounting gap; existing physical slot
   replacements and tail-cap metadata evictions have narrower, explicit scopes.
   Review the requirement matrix and finalize the report after missing evidence.
4. Keep the larger-budget throughput observation exploratory until repeated;
   the original18-run matrix remains a valid zero-hit result. GitHub publishing
   and course submission have not been performed.

Continuity: [active transcript](../../../chat-history/ASCEND_HOMEWORK_TRANSCRIPT.md).
