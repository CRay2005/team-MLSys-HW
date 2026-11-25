"""Prevent misleading performance aggregation and partial-suite acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("hw3_matrix_analysis", ROOT / "scripts/analyze_matrix.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class MatrixAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(workload={"batches": [
            dict(repeat=r, batch=b, input_ids=[[1] * n, [2] * n])
            for r, b, n in [(0, 0, 256), (0, 1, 256), (0, 2, 256), (1, 0, 264)]]},
            runs=[dict(seconds=t, outputs=[dict(output_ids=[1] * 16)] * 2) for t in (100, 1, 3, 2)])

    def test_cold_and_longer_revisits_are_not_pooled(self):
        groups = analysis.phases(self.result)
        self.assertEqual(set(groups), {"cold_256", "initial_256", "revisit_264"})
        self.assertEqual(groups["cold_256"]["seconds"], 100)
        self.assertEqual(groups["revisit_264"]["output_tokens_per_second"], 16)

    def test_throughput_is_tokens_over_time_not_mean_of_rates(self):
        group = analysis.phases(self.result)["initial_256"]
        self.assertEqual(group["output_tokens_per_second"], 64 / 4)
        self.assertEqual(group["batches"], 2)

    def test_invalid_duration_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                result = copy.deepcopy(self.result)
                result["runs"][1]["seconds"] = value
                with self.assertRaises(ValueError):
                    analysis.phases(result)

    def test_incomplete_suite_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text(json.dumps(dict(status="running", suite="matrix", jobs_completed=6)))
            with self.assertRaisesRegex(ValueError, "did not finish"):
                analysis.analyze(root)


if __name__ == "__main__":
    unittest.main()
