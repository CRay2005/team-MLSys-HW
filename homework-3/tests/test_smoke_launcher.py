"""Check packaging and fail-closed supervision without accessing an accelerator."""
import hashlib
import importlib.util
import json
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
