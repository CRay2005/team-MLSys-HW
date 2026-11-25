"""Observe existing cache operations without changing their fences or scheduling.

Completion latency is wall time from call entry to observed ACK, including queueing
and polling. It is not isolated DMA device time. Bytes count packed field payloads
and slot blobs, not allocator padding. Instrumentation adds logging overhead.
"""
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time


class Ledger:
    """Count a transfer batch only after every associated operation completes."""

    def __init__(self, emit):
        self.emit = emit
        self.pending = {}
        self.counter = 0
        self.lock = threading.RLock()

    def begin(self, kind, op_ids, payload_bytes, rows, requests, *, now_ns):
        ids = list(dict.fromkeys(map(int, op_ids)))
        if not ids:
            return None
        if payload_bytes < 0:
            raise ValueError("negative payload size")
        with self.lock:
            keys = [(kind, op) for op in ids]
            if any(key in self.pending for key in keys):
                raise ValueError("duplicate in-flight cache operation")
            self.counter += 1
            batch = dict(batch=self.counter, kind=kind, op_ids=ids, payload_bytes=payload_bytes,
                         rows=rows, requests=requests, started_ns=now_ns, remaining=set(keys))
            for key in keys:
                self.pending[key] = batch
            self.emit("transfer_begin", **{k: v for k, v in batch.items() if k != "remaining"})
            return batch

    def complete(self, kind, op_id, *, now_ns, basis="event_ack"):
        with self.lock:
            key = (kind, int(op_id))
            if key not in self.pending:
                raise ValueError(f"completion without submission: {key}")
            batch = self.pending.pop(key)
            batch["remaining"].remove(key)
            if not batch["remaining"]:
                self.emit("transfer_complete", batch=batch["batch"], kind=kind,
                          op_ids=batch["op_ids"], payload_bytes=batch["payload_bytes"],
                          observed_seconds=(now_ns - batch["started_ns"]) / 1e9,
                          completion_basis=basis)

    def fail(self, batch, error):
        if batch is None:
            return
        with self.lock:
            for key in list(batch["remaining"]):
                self.pending.pop(key, None)
            self.emit("transfer_failed", batch=batch["batch"], error=type(error).__name__)


class Observer:
    def __init__(self, rank):
        self.rank = int(rank)
        directory = Path(os.environ["HW3_TRACE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / f"rank-{rank}-pid-{os.getpid()}.jsonl").open("x", buffering=1)
        self.lock = threading.RLock()
        self.ledger = Ledger(self.emit)
        self.previous_stats = None
        self.last_idle_ns = 0
        self.emit("observer_start", timing="call-entry to ACK; includes queueing/polling", version=1)

    def emit(self, event, **values):
        with self.lock:
            self.file.write(json.dumps(dict(event=event, rank=self.rank, pid=os.getpid(),
                                           monotonic_ns=time.perf_counter_ns(), **values), separators=(",", ":")) + "\n")


_observer = None
_observer_lock = threading.Lock()


def observer(rank):
    global _observer
    with _observer_lock:
        if _observer is None:
            _observer = Observer(rank)
        elif _observer.rank != rank:
            raise ValueError("multiple scheduler ranks in one observer process")
        return _observer


def payload_bytes(layout, rows, blob_bytes, request_count):
    sizes = [sum(field.payload_bytes for field in group.fields) for group in layout.groups]
    return sum(sizes[int(row[0])] for row in rows) + blob_bytes * request_count


def instrument_executor(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trace = observer(self.attn_tp_rank)
        trace.emit("allocation", device_lcm_blocks=self.layout.num_lcm_blocks,
                   host_l2_bytes=0 if self.host_storage is None else self.host_storage.host_buffer_bytes,
                   host_snapshot_bytes=0 if self.snapshot_storage is None else self.snapshot_storage.host_buffer_bytes,
                   slot_blob_bytes=self.blob_bytes * (self.max_retracted_requests + 1),
                   snapshot_blocks=self.num_snapshot_pages - 1,
                   host_blocks=max(0, self.num_host_pages - 1),
                   groups=[dict(name=g.group_id, blocks_per_lcm=g.cache_blocks_per_lcm_block,
                                fields=[dict(name=f.field_id, bytes=f.payload_bytes) for f in g.fields])
                           for g in self.layout.groups])

    cls.__init__ = init
    for method, kind in [("_start_writing", "writeback"), ("_start_loading", "loadback"),
                         ("_start_snapshot_store", "snapshot"), ("_start_restore", "restore")]:
        original = getattr(cls, method)
        signature = inspect.signature(original)

        def make_wrapper(original, signature, kind):
            @functools.wraps(original)
            def wrapper(self, *args, **kwargs):
                values = signature.bind(self, *args, **kwargs).arguments
                trace = observer(self.attn_tp_rank)
                requests = []
                if kind in ("snapshot", "restore"):
                    ops = values["ops"]
                    ids = [op.op_id for op in ops]
                    requests = [dict(request_id=op.request_id, slot=op.request_pool_index,
                                     snapshot_slot=op.snapshot_slot) for op in ops]
                    if kind == "restore":
                        rows = [list(row) + [int(tier)] for tier, entries in values["rows_by_tier"].items() for row in entries]
                    else:
                        rows = [list(row) + [1] for row in values["transfers"]]
                    blobs = self.blob_bytes * len(ops)
                else:
                    ids = values["op_ids"]
                    rows = [list(row) + [0] for row in values["transfers"]]
                    blobs = 0
                batch = trace.ledger.begin(kind, ids, payload_bytes(self.layout, rows, 0, 0) + blobs,
                                           rows, requests, now_ns=time.perf_counter_ns())
                try:
                    return original(self, *args, **kwargs)
                except BaseException as exc:
                    trace.ledger.fail(batch, exc)
                    raise
            return wrapper
        setattr(cls, method, make_wrapper(original, signature, kind))

    original_poll = cls.poll_results
    kinds = {"WriteBackDoneEvent": "writeback", "LoadBackDoneEvent": "loadback",
             "SnapshotDoneEvent": "snapshot", "RestoreDoneEvent": "restore"}

    @functools.wraps(original_poll)
    def poll(self):
        results = original_poll(self)
        trace = observer(self.attn_tp_rank)
        for event in results:
            trace.ledger.complete(kinds[type(event).__name__], event.op_id, now_ns=time.perf_counter_ns())
        return results

    cls.poll_results = poll
    original_shutdown = cls.shutdown

    @functools.wraps(original_shutdown)
    def shutdown(self):
        original_shutdown(self)  # Existing implementation synchronizes the device before clearing ACKs.
        trace = observer(self.attn_tp_rank)
        for kind, op_id in list(trace.ledger.pending):
            trace.ledger.complete(kind, op_id, now_ns=time.perf_counter_ns(), basis="shutdown_device_synchronize")
        trace.emit("executor_shutdown", pending_operations=len(trace.ledger.pending))

    cls.shutdown = shutdown


def scheduler_snapshot(loop):
    scheduler = loop.scheduler
    return dict(device_total=loop._scheduler_cache_geometry.num_usable_pages,
                device_empty=scheduler.empty_lcm_blocks(), device_available=scheduler.available_lcm_blocks(),
                device_active=scheduler.active_lcm_blocks(), host_cached=scheduler.host_pool_cached_blocks(),
                host_pinned=scheduler.host_pool_pinned_blocks(), snapshot_free=scheduler.snapshot_pool_free_blocks(),
                tail_entries=scheduler.tail_entries(), tail_cap_evictions=scheduler.tail_cap_evictions(),
                waiting=scheduler.waiting_size(), retracted=scheduler.retracted_size(),
                decoding=scheduler.decoding_size(), prefilling=scheduler.prefilling_size())


def instrument_event_loop(cls):
    original = cls._get_forward_op

    @functools.wraps(original)
    def forward(self, plan):
        result = original(self, plan)
        trace = observer(self.attn_tp_rank)
        stats = scheduler_snapshot(self)
        if stats != trace.previous_stats:
            trace.emit("scheduler", **stats)
            trace.previous_stats = stats
        now = time.perf_counter_ns()
        if (result is None and not trace.ledger.pending and not any(stats[key] for key in
                ("device_active", "host_pinned", "waiting", "retracted", "decoding", "prefilling"))
                and now - trace.last_idle_ns >= 200_000_000):
            trace.emit("scheduler_quiescent", **stats, pending_operations=0)
            trace.last_idle_ns = now
        for op in plan.cache:
            kind = type(op).__name__
            if kind not in ("WriteBackOp", "LoadBackOp", "SnapshotOp", "RestoreOp"):
                continue
            data = dict(kind=kind, op_ids=list(op.op_ids), group_ids=list(op.group_ids),
                        source_blocks=list(op.src_pages), destination_blocks=list(op.dst_pages))
            if kind in ("WriteBackOp", "LoadBackOp", "RestoreOp"):
                data.update(content_hashes=list(op.content_hashes), page_offsets=list(op.page_offsets))
            if kind in ("SnapshotOp", "RestoreOp"):
                data.update(request_ids=list(op.request_ids), slots=list(op.request_pool_indices),
                            frontiers=[self.scheduler.request_token_size(rid) for rid in op.request_ids])
            trace.emit("cache_plan", **data)
        if result is not None:
            extend = result.num_extends()
            trace.emit("forward", request_ids=list(result.request_ids),
                       slots=list(result.request_pool_indices), input_lengths=list(result.input_lengths),
                       extend_count=extend, prefix_lengths=list(result.extend_prefix_lens),
                       replay_lengths=list(result.extend_replay_lens), block_tables=result.block_tables,
                       frontiers=[self.scheduler.request_token_size(rid) for rid in result.request_ids])
        return result

    cls._get_forward_op = forward
    original_close = cls.close

    @functools.wraps(original_close)
    def close(self):
        observer(self.attn_tp_rank).emit("scheduler_before_close", **scheduler_snapshot(self))
        return original_close(self)

    cls.close = close


def instrument_hooks(cls):
    original = cls.poll_ready_events

    @functools.wraps(original)
    def poll(self):
        results = original(self)
        if results:
            observer(self._attn_tp_rank).emit("rank_agreed_completions",
                operations=[dict(kind=type(e).__name__, op_id=e.op_id) for e in results])
        return results

    cls.poll_ready_events = poll


TARGETS = {
    "tokenspeed.runtime.cache.l2.executor": ("HostCacheExecutor", instrument_executor),
    "tokenspeed.runtime.engine.event_loop": ("EventLoop", instrument_event_loop),
    "tokenspeed.runtime.engine.cache_hooks": ("CacheOpHooks", instrument_hooks),
}


class Loader:
    def __init__(self, original, target):
        self.original = original
        self.target = target

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        name, instrument = TARGETS[self.target]
        instrument(getattr(module, name))


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None:
            spec.loader = Loader(spec.loader, fullname)
        return spec


def install():
    if any(name in sys.modules for name in TARGETS):
        raise RuntimeError("HW3 observation must be installed before runtime imports")
    if not any(isinstance(finder, Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, Finder())
