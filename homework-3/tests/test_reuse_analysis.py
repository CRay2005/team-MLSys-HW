"""Require completed copies, distinct prefixes and unoverwritten Host slots."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("hw3_reuse_analysis", ROOT / "scripts/analyze_reuse.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


def forward(t, rid, block, prefix=0):
    return dict(event="forward", monotonic_ns=t, request_ids=[rid], extend_count=1,
                prefix_lengths=[prefix], block_tables={"full": [[block]]})


def plan(t, kind, op, source, destination):
    return dict(event="cache_plan", monotonic_ns=t, kind=kind, op_ids=[op], group_ids=[[0]],
                source_blocks=[[source]], destination_blocks=[[destination]])


def ack(t, kind, op):
    return dict(event="transfer_complete", monotonic_ns=t, kind=kind, op_ids=[op])


class ReuseAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.events = [dict(event="allocation", monotonic_ns=0, groups=[dict(name="full")]),
                       forward(1, "a", 1), plan(2, "WriteBackOp", 10, 1, 7), ack(3, "writeback", 10),
                       forward(4, "b", 1), plan(5, "LoadBackOp", 11, 7, 2),
                       forward(6, "a2", 2, prefix=128), ack(7, "loadback", 11)]
        self.markers = dict(a=1, b=2, a2=1)

    def test_load_ack_can_follow_cpu_forward_plan(self):
        rows = reuse.witnesses(self.events, self.markers)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["original_device_block"], 1)
        self.assertEqual(rows[0]["restored_device_block"], 2)
        self.assertGreater(rows[0]["loadback_ack_ns"], rows[0]["resumed_forward_plan_ns"])

    def test_missing_load_ack_is_not_evidence(self):
        self.assertEqual(reuse.witnesses(self.events[:-1], self.markers), [])

    def test_shared_prefix_owner_change_is_not_evidence(self):
        self.markers["b"] = 1
        self.assertEqual(reuse.witnesses(self.events, self.markers), [])

    def test_host_overwrite_invalidates_old_payload_identity(self):
        events = self.events[:5] + [plan(4.1, "WriteBackOp", 12, 1, 7), ack(4.2, "writeback", 12)] + self.events[5:]
        self.assertEqual(reuse.witnesses(events, self.markers), [])


if __name__ == "__main__":
    unittest.main()
