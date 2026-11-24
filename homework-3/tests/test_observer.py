"""Exercise completion accounting with real ledger transitions and event data."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load("hw3_observer_test", ROOT / "runtime/runtime_observer.py")
summary = load("hw3_summary_test", ROOT / "scripts/summarize_trace.py")


class ObserverTests(unittest.TestCase):
    def test_batch_counts_bytes_once_after_all_acks(self):
        events = []
        ledger = observer.Ledger(lambda event, **values: events.append(dict(event=event, **values)))
        ledger.begin("restore", [1, 2], 4096, [], [], now_ns=1_000_000_000)
        ledger.complete("restore", 2, now_ns=2_000_000_000)
        self.assertEqual([e["event"] for e in events], ["transfer_begin"])
        ledger.complete("restore", 1, now_ns=3_000_000_000)
        self.assertEqual(events[-1]["payload_bytes"], 4096)
        self.assertEqual(events[-1]["observed_seconds"], 2)
        self.assertFalse(ledger.pending)
        with self.assertRaises(ValueError):
            ledger.complete("restore", 1, now_ns=4_000_000_000)

    def test_equal_ids_in_different_operation_kinds_are_distinct(self):
        ledger = observer.Ledger(lambda *args, **kwargs: None)
        for kind in ("snapshot", "restore"):
            ledger.begin(kind, [1], 4, [], [], now_ns=0)
        ledger.complete("snapshot", 1, now_ns=1)
        self.assertEqual(set(ledger.pending), {("restore", 1)})

    def test_duplicate_inflight_is_rejected_and_failure_drops_batch(self):
        events = []
        ledger = observer.Ledger(lambda event, **values: events.append(event))
        batch = ledger.begin("writeback", [3, 4], 8, [], [], now_ns=0)
        with self.assertRaises(ValueError):
            ledger.begin("writeback", [4], 1, [], [], now_ns=0)
        ledger.fail(batch, RuntimeError("copy failed"))
        self.assertFalse(ledger.pending)
        self.assertNotIn("transfer_complete", events)

    def test_summary_rejects_submission_without_completion(self):
        with self.assertRaisesRegex(ValueError, "lack observed completion"):
            summary.summarize([{"event": "observer_start", "rank": 0},
                               {"event": "transfer_begin", "batch": 1}])

    def test_summary_rejects_mismatched_restoration_frontier(self):
        rows = [{"event": "observer_start", "rank": 0},
                {"event": "cache_plan", "kind": "SnapshotOp", "request_ids": ["a"], "frontiers": [128]},
                {"event": "cache_plan", "kind": "RestoreOp", "request_ids": ["a"], "frontiers": [127]},
                {"event": "scheduler_before_close", **dict.fromkeys(
                    ["device_active", "host_cached", "host_pinned", "retracted", "tail_cap_evictions"], 0)}]
        with self.assertRaisesRegex(ValueError, "frontier mismatch"):
            summary.summarize(rows)


if __name__ == "__main__":
    unittest.main()
