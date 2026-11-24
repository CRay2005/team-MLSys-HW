# HW3 runtime observation

`launch_smoke.py --suite trace` enables the observer in every spawned rank through
an explicit `HW3_TRACE_DIR` and isolated `homework-3/runtime` import path. Runtime
source files, scheduler decisions, stream fences and completion queues are
unchanged. Post-import wrappers call original methods with arguments intact.
Ordinary smoke runs leave observation disabled. Observer hashes and enabled state
are recorded and checked so instrumented/uninstrumented runs cannot be paired.

Rank-local copied rows determine payload bytes: sum all group-field payloads plus
slot blobs; exclude allocator padding. A batch completes only after every
operation ID receives an ACK. A successful existing shutdown device synchronization
can account for final discarded ACKs, recorded as a separate completion basis.
Unknown/duplicate completions and unfinished submissions fail validation.

Latency measures host-observed method-entry to ACK time, including queueing,
polling and observation overhead. It is not isolated DMA device time. Overlapping
durations must not be summed as end-to-end wall time. Request/batch timings stay
in `result.json`; performance comparisons must disclose observation overhead.

Control-plane records expose device active/empty/available LCM blocks, Host cached
and pinned entries, snapshot free LCM blocks, and waiting/retracted counts. Host
cached entries are per-group index entries, not LCM blocks: do not multiply them
by an LCM byte size. Report allocation bytes and occupancy counts separately.
Per-forward block tables, prefix positions, replay lengths and snapshot/restore
frontiers are retained. The checker rejects unequal stored/restored frontiers.
Replica-agreed completion IDs are separate from rank-local ACKs.

Host destinations assigned different content keys count physical replacements,
**not all logical evictions**. First-admission prefix hits/misses include every
tier; Host load-back operation counts are separate. Device owner changes can also
reflect prefix sharing and do not alone prove reclaimed private blocks. These
limits remain explicit. Pressure, recovery and stable-resource experiments are
still required before marking HW3 complete.
