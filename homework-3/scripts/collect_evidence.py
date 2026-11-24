"""Download only terminal HW3 runs, verifying every file before extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile


REMOTE = '''import hashlib, io, json, pathlib, sys, tarfile
root = pathlib.Path.home() / "MLSys-HW" / LABEL
status = json.loads((root / "status.json").read_text())
assert status["status"] in {"suite_passed", "smoke_passed", "failed", "blocked"}, "Run is not terminal"
paths = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()
         and (p.relative_to(root).parts[0] == "results" or p.parent == root and p.suffix in {".json", ".log"})]
payload = {str(p.relative_to(root)): p.read_bytes() for p in sorted(paths)}
manifest = {"run": LABEL, "files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
payload["evidence-manifest.json"] = (json.dumps(manifest, indent=2) + "\\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tar:
    for name, data in payload.items():
        member = tarfile.TarInfo(name); member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
'''


def verify_archive(archive):
    with tarfile.open(archive) as tar:
        names = []
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe evidence archive entry")
            names.append(member.name)
        if len(names) != len(set(names)):
            raise ValueError("duplicate evidence entry")
        manifest = json.load(tar.extractfile("evidence-manifest.json"))
        if set(names) != set(manifest["files"]) | {"evidence-manifest.json"}:
            raise ValueError("evidence manifest/file set differs")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != expected:
                raise ValueError(f"evidence checksum mismatch: {name}")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host.startswith("-") or any(c.isspace() for c in args.host):
        parser.error("invalid SSH destination")
    if not re.fullmatch(r"hw3-smoke-\d{8}T\d{6}Z-[a-f0-9]{8}", args.run):
        parser.error("invalid generated run label")
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10",
           "-o", "ServerAliveCountMax=2", "-i", str(Path.home() / ".ssh/id_ed25519"), args.host, "python3 -"]
    # Streaming archive avoids creating redundant recovery archives on the server.
    with tempfile.TemporaryDirectory(prefix="hw3-evidence-", dir=args.output.parent) as tmp:
        archive = Path(tmp) / "evidence.tar.gz"
        with archive.open("wb") as file:
            subprocess.run(ssh, input=(f"LABEL={args.run!r}\n" + REMOTE).encode(), stdout=file,
                           check=True, timeout=180)
        manifest = verify_archive(archive)
        if manifest["run"] != args.run:
            raise ValueError("unexpected evidence run identity")
        args.output.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(args.output, filter="data")
    print(json.dumps(dict(run=args.run, verified_files=len(manifest["files"]), output=str(args.output))))


if __name__ == "__main__":
    main()
