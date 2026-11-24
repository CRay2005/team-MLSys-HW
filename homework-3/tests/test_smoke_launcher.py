"""Check packaging and fail-closed supervision without accessing an accelerator."""
import hashlib
import importlib.util
import json
import socket
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("launch_smoke")
server = load("server_smoke")


class SmokeLauncherTests(unittest.TestCase):
    def test_queue_requires_successful_predecessor(self):
        label = "hw3-smoke-20261124T111450Z-b3d422e4"
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "BASE", Path(tmp)):
            root = Path(tmp) / label
            root.mkdir()
            (root / "status.json").write_text(json.dumps(dict(status="suite_passed")))
            server.wait_for_run(label, timeout=0)
            for status in ("failed", "blocked"):
                (root / "status.json").write_text(json.dumps(dict(status=status)))
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    server.wait_for_run(label, timeout=0)
            (root / "status.json").write_text(json.dumps(dict(status="running")))
            with self.assertRaises(TimeoutError):
                server.wait_for_run(label, timeout=0)
            with self.assertRaises(ValueError):
                server.wait_for_run("../outside", timeout=0)

    def test_performance_matrix_uses_normal_scheduling_and_matched_references(self):
        jobs = server.experiment_jobs("matrix")
        self.assertEqual(len(jobs), 18)
        by_name = {job["name"]: job for job in jobs}
        self.assertEqual(len(by_name), 18)
        self.assertEqual({job["pattern"] for job in jobs}, {"high", "low"})
        self.assertEqual({job["repetition"] for job in jobs}, {0, 1, 2})
        self.assertEqual({job["ratio"] for job in jobs if job["mode"] == "on"}, {"0.5", "1"})
        for job in jobs:
            self.assertEqual(job["force"], "0")
            reference = by_name[job["reference"]]
            self.assertEqual(reference["mode"], "off")
            for key in ("pattern", "working_set", "cycles", "repetition"):
                self.assertEqual(job[key], reference[key])

    def test_stress_extends_cycles_and_tp2_stays_on_one_card(self):
        self.assertTrue(all(job["cycles"] == 8 and job["working_set"] == 8 for job in server.experiment_jobs("stress")))
        self.assertEqual(server.physical_devices(2, 2), [2, 3])
        with self.assertRaises(ValueError):
            server.physical_devices(1, 2)

    def test_occupied_port_is_not_reused(self):
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 0))
            sock.listen()
            self.assertFalse(server.ports_available([sock.getsockname()[1]]))

    def test_choose_ports_skips_occupied_range(self):
        with patch.object(server, "ports_available", side_effect=[False, True]), patch.dict(server.os.environ):
            port, ports = server.choose_ports()
            self.assertEqual(port, 31893)
            self.assertEqual(ports, list(range(25064, 25096)))
            self.assertEqual(server.os.environ["HCCL_NPU_SOCKET_PORT_RANGE"], "25064-25095")

    def test_archive_hashes_and_safe_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = launcher.build_bundle(Path(tmp))
            with tarfile.open(archive) as tar:
                manifest = json.load(tar.extractfile("deployment-manifest.json"))
                self.assertEqual(set(tar.getnames()), set(launcher.FILES) | {"deployment-manifest.json"})
                for name, digest in manifest["files"].items():
                    self.assertTrue(tar.getmember(name).isfile())
                    self.assertNotIn("..", Path(name).parts)
                    self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(), digest)
                self.assertEqual(len(manifest["expected_runtime_trees"]), 3)

    def test_failed_command_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "ROOT", Path(tmp)):
            (Path(tmp) / "results").mkdir()
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                server.bounded([sys.executable, "-c", "raise SystemExit(7)"], "failure", 5)

    def test_timeout_terminates_own_process_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "ROOT", Path(tmp)):
            (Path(tmp) / "results").mkdir()
            with self.assertRaises(subprocess.TimeoutExpired):
                server.bounded([sys.executable, "-c", "import time; time.sleep(30)"], "timeout", 0.1)

    def test_unknown_npu_output_refuses_launch(self):
        result = subprocess.CompletedProcess([], 0, stdout="unexpected npu-smi format")
        with patch.object(server.subprocess, "run", return_value=result), self.assertRaises(ValueError):
            server.assert_free(1)

    def test_occupied_device_refuses_launch(self):
        result = subprocess.CompletedProcess([], 0, stdout=(
            "| NPU Chip | Process id | Process name | Process memory(MB) |\n"
            "| 0 1 | 1234 | python | 2000 |\n"))
        with patch.object(server.subprocess, "run", return_value=result), self.assertRaisesRegex(RuntimeError, "occupied"):
            server.assert_free(1)


if __name__ == "__main__":
    unittest.main()
