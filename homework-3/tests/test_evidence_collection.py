"""Evidence integrity must be checked before any extraction."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("hw3_collect", Path(__file__).resolve().parents[1] / "scripts/collect_evidence.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class EvidenceTests(unittest.TestCase):
    def test_verified_file_and_tampered_payload(self):
        for payload in (b"original", b"tampered"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "test.tar.gz"
                manifest = dict(run="example", files={"results/item.json": hashlib.sha256(b"original").hexdigest()})
                with tarfile.open(archive, "w:gz") as tar:
                    for name, data in [("results/item.json", payload), ("evidence-manifest.json", json.dumps(manifest).encode())]:
                        entry = tarfile.TarInfo(name); entry.size = len(data)
                        tar.addfile(entry, io.BytesIO(data))
                if payload == b"original":
                    self.assertEqual(collect.verify_archive(archive), manifest)
                else:
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        collect.verify_archive(archive)

    def test_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "test.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.addfile(tarfile.TarInfo("../outside"), io.BytesIO(b""))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                collect.verify_archive(archive)


if __name__ == "__main__":
    unittest.main()
