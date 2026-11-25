import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_offload import read_trace, workload


class OffloadRunnerTest(unittest.TestCase):
    def test_workloads_match_lengths_and_only_high_repeats_prefixes(self):
        high = workload("high", 192, 6, 16)
        low = workload("low", 192, 6, 16)

        def lengths(trace):
            return [(list(map(len, batch)), count) for batch, count in trace]

        self.assertEqual(lengths(high), lengths(low))
        self.assertEqual(high[3][0][0][:-1], high[0][0][0])
        self.assertEqual(high[5][0][0][:-1], high[1][0][0])
        self.assertEqual(len({batch[0][0] for batch, _ in low}), len(low))

    def test_reads_real_logger_format_for_each_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            lines = []
            for rank in (0, 1):
                prefix = f"\x1b[38;20m[2026-11-22 09:09:49,123 ATTN TP RANK {rank}] - INFO - "
                lines.append(
                    prefix
                    + f"[host-cache-pools] rank={rank} l2_bytes=512 snapshot_bytes=0 "
                    "blob_arena_bytes=0 lcm_block_bytes=64 (executor.py:326)\x1b[0m"
                )
                lines.append(
                    prefix
                    + f"[host-cache-round] rank={rank} device_total=12 device_used=8 "
                    "device_empty=4 device_reclaimable=8 host_free=4 snapshot_free=0 "
                    "host_hits=1 host_misses=3 host_evictions=0 prefill=[(192, 0, 1)] "
                    "(event_loop.py:1148)\x1b[0m"
                )
                lines.append(
                    prefix
                    + f"[host-cache-copy] rank={rank} kind=LoadBackDone operations=1 "
                    "bytes=256 completion_ms=0.706 ids=[3] (executor.py:1186)\x1b[0m"
                )
            path.write_text("\n".join(lines))
            pools, traces = read_trace(path, 2)
        self.assertEqual(pools[1]["l2_bytes"], 512)
        self.assertEqual(traces[0]["rounds"][0]["prefill"], [(192, 0, 1)])
        self.assertEqual(traces[1]["copies"][0]["ids"], [3])
        self.assertEqual(traces[1]["copies"][0]["completion_ms"], 0.706)


if __name__ == "__main__":
    unittest.main()
