"""Local behavioral tests; no claim of NPU execution."""

import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("hw3", Path(__file__).resolve().parents[1] / "scripts/run_offloading.py")
hw3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hw3)


def config(**changes):
    args = dict(model="/models/0.8B", tp=1, devices=[1], memory_fraction=0.12,
                port=31893, offloading="on", host_ratio=1.0, snapshot_ratio=1.0,
                force_interval=0, device_tokens=1024, dtype="bfloat16", head_dtype="native")
    args.update(changes)
    return hw3.build_config(**args)


def result_pair():
    workload = hw3.make_workload(list(range(256)), vocab_size=1000, pattern="high", repeats=1)
    base = dict(status="generation_completed_unverified", config=config(), workload=workload,
                workload_sha256=hw3.digest(workload), model_config_sha256="config",
                model_manifest_sha256="manifest", source_fingerprints={"python": "hash"},
                runtime_versions={"torch": "test"}, physical_devices=[1], runs=[])
    for batch in workload["batches"]:
        base["runs"].append(dict(repeat=batch["repeat"], batch=batch["batch"], outputs=[
            {"output_ids": [s] * 16, "meta_info": {"prompt_tokens": 256,
             "completion_tokens": 16, "cached_tokens": 128}} for s in batch["sessions"]]))
    reference = copy.deepcopy(base)
    reference["config"] = config(offloading="off", host_ratio=0, snapshot_ratio=0)
    return reference, base


class OffloadTests(unittest.TestCase):
    def test_off_disables_both_host_pools_but_keeps_device_prefix_cache(self):
        c = config(offloading="off", host_ratio=0, snapshot_ratio=0)
        self.assertTrue(c["disable_kvstore"])
        self.assertEqual(c["retraction_snapshot_ratio"], 0)
        self.assertTrue(c["enable_prefix_caching"])

    def test_hidden_host_pool_or_force_in_off_rejected(self):
        for changes in ({"host_ratio": 1}, {"snapshot_ratio": 1}, {"force_interval": 2}):
            args = dict(offloading="off", host_ratio=0, snapshot_ratio=0)
            args.update(changes)
            with self.assertRaises(ValueError):
                config(**args)

    def test_invalid_budgets_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                config(host_ratio=value)
        with self.assertRaises(ValueError):
            config(snapshot_ratio=0)

    def test_precision_and_device_budget_preserved(self):
        c = config(dtype="float16", head_dtype="float32", device_tokens=2048)
        self.assertEqual(c["kv_cache_dtype"], "float16")
        self.assertEqual(c["lm_head_compute_dtype"], "float32")
        self.assertEqual(c["max_total_tokens"], 2048)
        self.assertEqual(c["mamba_ssm_dtype"], "float32")

    def test_reuse_patterns_equal_size_different_reuse(self):
        high, low = [hw3.make_workload([10] * 256, vocab_size=1000, pattern=p, repeats=3)
                     for p in ("high", "low")]
        markers = lambda w: [ids[0] for b in w["batches"] for ids in b["input_ids"]]
        self.assertEqual(len(markers(high)), 24)
        self.assertEqual(len(set(markers(high))), 2)
        self.assertEqual(len(set(markers(low))), 24)
        self.assertNotEqual(high["batches"][0]["sessions"], high["batches"][1]["sessions"])

    def test_invalid_seed_rejected(self):
        for seed in ([], [True], [-1], [1000], [1] * 500):
            with self.assertRaises(ValueError):
                hw3.make_workload(seed, vocab_size=1000, pattern="high", repeats=1)

    def test_agreement_never_claims_offload_proof(self):
        result = hw3.compare_runs(*result_pair())
        self.assertEqual(result["exact_token_agreement_requests"], 8)
        self.assertFalse(result["offload_lifecycle_verified"])

    def test_every_request_compared(self):
        for batch in range(4):
            for request in range(2):
                a, b = result_pair()
                b["runs"][batch]["outputs"][request]["output_ids"][-1] += 1
                with self.assertRaisesRegex(ValueError, "token IDs differ"):
                    hw3.compare_runs(a, b)

    def test_incomplete_wrong_order_or_source_rejected(self):
        for mutation in (lambda b: b["runs"].pop(),
                         lambda b: b["runs"].reverse(),
                         lambda b: b.update(status="failed"),
                         lambda b: b.update(model_manifest_sha256="different"),
                         lambda b: b["config"].update(max_total_tokens=2048),
                         lambda b: b["workload"].update(new_tokens=10)):
            a, b = result_pair()
            mutation(b)
            with self.assertRaises(ValueError):
                hw3.compare_runs(a, b)

    def test_malformed_generation_rejected(self):
        a, b = result_pair()
        b["runs"][0]["outputs"][0]["meta_info"]["cached_tokens"] = 300
        with self.assertRaises(ValueError):
            hw3.compare_runs(a, b)


if __name__ == "__main__":
    unittest.main()
