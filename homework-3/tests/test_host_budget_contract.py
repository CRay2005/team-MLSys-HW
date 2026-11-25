"""Exercise the real upstream pool-sizing implementation without accelerator mocks."""

import importlib.util
from pathlib import Path
import sys
import unittest

SOURCE = (Path(__file__).resolve().parents[2] / "homework-1/tokenspeed/python"
          / "tokenspeed/runtime/cache/l2/sizing.py")
SPEC = importlib.util.spec_from_file_location("hw3_pool_sizing", SOURCE)
sizing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sizing
SPEC.loader.exec_module(sizing)


class HostBudgetContractTests(unittest.TestCase):
    def resolve(self, request, *, l2):
        return sizing.resolve_retraction_pool(request, l2_tier=l2,
                                              device_lcm_blocks=100,
                                              host_lcm_block_bytes=1_000_000)

    def test_disabled_kvstore_alone_still_allocates_snapshot_pool(self):
        request = sizing.RetractionPoolRequest(host_gb=0, ratio=None,
                                               max_retracted_requests=2,
                                               tail_lcm_blocks_per_request=3)
        self.assertEqual(self.resolve(request, l2=False).lcm_blocks, 10)
        self.assertEqual(self.resolve(request, l2=True).lcm_blocks, 6)

    def test_explicit_zero_disables_snapshots_with_either_l2_setting(self):
        request = sizing.RetractionPoolRequest(host_gb=0, ratio=0,
                                               max_retracted_requests=0,
                                               tail_lcm_blocks_per_request=3)
        for l2 in (True, False):
            actual = self.resolve(request, l2=l2)
            self.assertEqual((actual.lcm_blocks, actual.max_retracted_requests), (0, 0))

    def test_snapshot_ratio_scales_device_capacity_not_l2_capacity(self):
        for ratio, expected in ((0.5, 50), (1, 100), (2, 200)):
            request = sizing.RetractionPoolRequest(host_gb=0, ratio=ratio,
                                                   max_retracted_requests=2,
                                                   tail_lcm_blocks_per_request=3)
            for l2 in (True, False):
                self.assertEqual(self.resolve(request, l2=l2).lcm_blocks, expected)

    def test_absolute_host_budget_overrides_zero_ratio(self):
        request = sizing.RetractionPoolRequest(host_gb=0.025, ratio=0,
                                               max_retracted_requests=2,
                                               tail_lcm_blocks_per_request=3)
        self.assertFalse(request.disabled)
        self.assertEqual(self.resolve(request, l2=False).lcm_blocks, 25)

    def test_insufficient_budget_and_missing_slot_capacity_fail(self):
        for request in (sizing.RetractionPoolRequest(0, 0.001, 2, 3),
                        sizing.RetractionPoolRequest(0, 1, 0, 3)):
            with self.assertRaises(ValueError):
                self.resolve(request, l2=True)


if __name__ == "__main__":
    unittest.main()
