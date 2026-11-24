"""Local behavioral tests; no claim of NPU execution."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
                harness_fingerprints=hw3.harness_fingerprints(),
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

    def test_empty_or_duplicate_workload_cannot_pass(self):
        for empty in (True, False):
            a, b = result_pair()
            for result in (a, b):
                if empty:
                    result["workload"]["batches"] = []
                    result["runs"] = []
                else:
                    result["workload"]["batches"].append(copy.deepcopy(result["workload"]["batches"][0]))
                    result["runs"].append(copy.deepcopy(result["runs"][0]))
                result["workload_sha256"] = hw3.digest(result["workload"])
            with self.assertRaises(ValueError):
                hw3.compare_runs(a, b)

    def test_runner_or_helper_change_invalidates_pair(self):
        for name in hw3.harness_fingerprints():
            a, b = result_pair()
            b["harness_fingerprints"][name] = "changed"
            with self.assertRaisesRegex(ValueError, "harness_fingerprints"):
                hw3.compare_runs(a, b)

    def test_absolute_snapshot_budget_cannot_hide_in_off_reference(self):
        a, b = result_pair()
        # Production sizing gives positive absolute bytes precedence over ratio 0.
        for result in (a, b):
            result["config"]["retraction_snapshot_host_gb"] = 0.025
        with self.assertRaisesRegex(ValueError, "retraction_snapshot_host_gb"):
            hw3.compare_runs(a, b)

    def test_malformed_token_list_or_count_rejected(self):
        for mutation in (lambda o: o.update(output_ids="0" * 16),
                         lambda o: o["meta_info"].update(completion_tokens=16.0)):
            a, b = result_pair()
            mutation(b["runs"][0]["outputs"][0])
            with self.assertRaises(ValueError):
                hw3.compare_runs(a, b)

    def test_failed_save_keeps_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            hw3.save_result(path, {"status": "running", "runs": [1]})
            with self.assertRaises(ValueError):
                hw3.save_result(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "running", "runs": [1]})
            hw3.save_result(path, {"status": "completed", "runs": [1, 2]})
            self.assertEqual(json.loads(path.read_text())["runs"], [1, 2])
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_runtime_must_use_the_fingerprinted_source_trees(self):
        base = hw3.ROOT / "homework-1/tokenspeed"
        paths = {"tokenspeed": [str(base / "python/tokenspeed")],
                 "tokenspeed_kernel": [str(base / "tokenspeed-kernel/python/tokenspeed_kernel")],
                 "tokenspeed_kernel_npu": [str(base / "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu")]}
        hw3.validate_source_locations(paths)
        for name in paths:
            for wrong in ([], ["/different/installation"], paths[name] * 2):
                modified = dict(paths)
                modified[name] = wrong
                with self.assertRaisesRegex(ValueError, name):
                    hw3.validate_source_locations(modified)

    def cli(self, folder, action):
        return subprocess.run([
            sys.executable, str(SPEC.origin), "--action", action,
            "--model", str(folder / "missing-model"),
            "--model-config", str(hw3.ROOT / "homework-2/metadata/Qwen3.5-0.8B/config.json"),
            "--seed-cases", str(hw3.ROOT / "homework-2/cases.json"),
            "--tp", "1", "--physical-devices", "1", "--memory-fraction", "0.12",
            "--device-tokens", "1024", "--port", "31893", "--offloading", "off",
            "--host-ratio", "0", "--snapshot-ratio", "0", "--force-interval", "0",
            "--dtype", "bfloat16", "--head-dtype", "native", "--pattern", "high",
            "--repeats", "1", "--output-dir", str(folder / "result")],
            capture_output=True, text=True, timeout=10)

    def test_plan_cli_does_not_run_and_refuses_result_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            process = self.cli(folder, "plan")
            self.assertEqual(process.returncode, 0, process.stderr)
            result_path = folder / "result/result.json"
            before = result_path.read_bytes()
            result = json.loads(before)
            self.assertEqual(result["status"], "planned_not_run")
            self.assertEqual(result["runs"], [])
            self.assertEqual(len(result["workload"]["batches"]), 4)
            self.assertNotEqual(self.cli(folder, "plan").returncode, 0)
            self.assertEqual(result_path.read_bytes(), before)

    def test_missing_model_marks_run_failed_before_npu_import(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            process = self.cli(folder, "run")
            self.assertNotEqual(process.returncode, 0)
            result = json.loads((folder / "result/result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("source-manifest.json", result["error"])
            self.assertEqual(result["runs"], [])


if __name__ == "__main__":
    unittest.main()
