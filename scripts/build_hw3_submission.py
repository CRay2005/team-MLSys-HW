#!/usr/bin/env python3
"""Build the HW3 PDF and checksummed ZIP offline from a staged Git checkout.

Requires the initialized TokenSpeed submodule (with its pinned base object),
Pandoc, headless Chromium, Poppler and CJK fonts. Stage edits you want included;
unstaged and untracked homework files are not read. This script never uploads.
"""

import argparse
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

# Import checkout-local helpers without writing __pycache__ into the checkout.
sys.dont_write_bytecode = True
from build_submissions import (
    BASE, HW1_SHA, PREFIX, json_bytes, pdf_ast, relative_links, run, safe_name, sha256, write_file,
)

ROOT = Path(__file__).resolve().parents[1]
HW = "homework-3"
REPORT = f"{HW}/report.md"
PDF = f"{PREFIX}_HW3_Report.pdf"
ZIP = f"{PREFIX}_HW3_Submission.zip"
SUMS = "SHA256SUMS"
PATCHES = (
    ("homework-1/patches/ascend-qwen3.5.patch", HW1_SHA),
    ("homework-2/patches/hccl-torch29.patch", "76071724b56e1f7922dd3fd9a4c2c5b359f3422252ac10c74daa5e5d0f2350f6"),
    (f"{HW}/patches/ascend-host-cache.patch", "a86d51527bd8b8adb2171d48397abea40b4f3119c87b5268ae55132aefe79655"),
)
HW2_VERIFICATION = "homework-2/patches/patch-verification.json"
PREREQUISITES = {
    *(name for name, _ in PATCHES), HW2_VERIFICATION,
    "homework-1/patches/manifest.json", "homework-2/patches/manifest.json",
    "homework-1/scripts/download_model.py", "homework-1/scripts/npu-constraints.txt",
    "homework-2/README.md", "homework-2/cases.json",
    *(f"homework-2/scripts/{name}.py" for name in (
        "run_model", "summarize_large_run", "summarize_benchmarks", "hccl_smoke",
        "compare_tp", "resume_weights", "npu_occupancy", "summarize_streaming",
    )),
    *(f"homework-2/metadata/Qwen3.5-{size}/{name}.json"
      for size in ("0.8B", "4B") for name in ("config", "source-manifest")),
}
REQUIRED = PREREQUISITES | {
    REPORT, f"{HW}/assignment.md", f"{HW}/README.md", f"{HW}/scripts/run_offload.py",
    f"{HW}/results/ascend-runs.tar.gz", f"{HW}/results/ascend-comparisons.json",
    *(f"{HW}/results/tp1-pr3/{name}.json" for name in ("capacity", "matrix", "stress")),
}
MATERIAL_DIRS = (HW,)
MATERIAL_SUFFIXES = {
    ".md", ".py", ".sh", ".json", ".jsonl", ".txt", ".log", ".out", ".csv", ".tsv", ".gz",
    ".patch", ".diff", ".sha256", ".yaml", ".yml", ".toml", ".svg", ".png", ".jpg", ".jpeg",
}
PRIVATE_DIRS = {".git", ".venv", "__pycache__", ".local-review", ".ssh", ".aws", ".cache"}
PRIVATE_SUFFIXES = {
    ".safetensors", ".gguf", ".ggml", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".pem", ".key", ".p12",
}
PRIVATE_NAMES = re.compile(r"credential|password|secret|campus.auth|^id_(rsa|dsa|ecdsa|ed25519)", re.I)


def git(*args, cwd=None, data=None):
    return run(["git", "--no-replace-objects", "-c", "protocol.allow=never", *args],
               cwd=ROOT if cwd is None else cwd, data=data)


def private_path(path):
    return (bool(PRIVATE_DIRS.intersection(part.lower() for part in path.parts))
            or bool(PRIVATE_SUFFIXES.intersection(suffix.lower() for suffix in path.suffixes))
            or any(part.lower().startswith(".env") or PRIVATE_NAMES.search(part) for part in path.parts)
            or path.name.lower() in {".netrc", ".npmrc", ".pypirc", "known_hosts", "authorized_keys"})


def stage_materials(stage):
    copied = set()
    entries = git("ls-files", "--stage", "-z", "--", *MATERIAL_DIRS, *sorted(PREREQUISITES))
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, oid, index_stage = metadata.decode("ascii").split()
        name = os.fsdecode(raw_name)
        path = PurePosixPath(name)
        if (private_path(path) or {"models", "weights", "wheels", ".source"}.intersection(path.parts)
                or (name not in PREREQUISITES and path.suffix.lower() not in MATERIAL_SUFFIXES)):
            continue
        if index_stage != "0" or mode not in ("100644", "100755"):
            raise ValueError(f"expected a resolved, regular tracked file: {name}")
        write_file(stage, name, git("cat-file", "blob", oid), int(mode, 8) & 0o777)
        copied.add(name)
    missing = REQUIRED - copied
    if missing:
        raise ValueError("stage the required HW3 and prerequisite files first: " + ", ".join(sorted(missing)))


def source_inventory(source):
    inventory = {}
    for path in source.rglob("*"):
        name = safe_name(path.relative_to(source).as_posix())
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or private_path(PurePosixPath(name)):
            raise ValueError(f"unexpected source path: {name}")
        inventory[name] = {"mode": "100755" if path.stat().st_mode & 0o111 else "100644",
                           "sha256": sha256(path.read_bytes())}
    return inventory


def reconstruct(stage, work):
    submodule = ROOT / "homework-1/tokenspeed"
    if not (submodule / ".git").exists():
        raise ValueError(f"initialize {submodule.relative_to(ROOT)} before building (no fetch is performed)")
    source = stage / HW / ".source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", BASE, cwd=submodule)), mode="r:") as archive:
        for member in archive:
            name = safe_name(member.name)
            if member.isdir():
                (source / name).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or private_path(PurePosixPath(name)) or (source / name).exists():
                raise ValueError(f"unexpected entry in pinned source archive: {name}")
            with archive.extractfile(member) as stream:
                write_file(source, name, stream.read(), 0o755 if member.mode & 0o111 else 0o644)

    verify = json.loads((stage / HW2_VERIFICATION).read_bytes())
    if (verify.get("passed") is not True or verify["base_commit"] != BASE
            or verify["hw1_sha256"] != HW1_SHA or verify["hw2_sha256"] != PATCHES[1][1]):
        raise ValueError("unexpected retained HW2 patch verification record")
    # An explicit nonexistent Git dir keeps apply independent of any enclosing checkout.
    apply = (f"--git-dir={work / 'no-git'}", f"--work-tree={source}",
             "-c", "apply.ignoreWhitespace=no", "apply", "--whitespace=error")
    for index, (name, expected) in enumerate(PATCHES, start=1):
        patch = (stage / name).read_bytes()
        if sha256(patch) != expected:
            raise ValueError(f"retained patch checksum differs: {name}")
        git(*apply, "--check", "-", cwd=source, data=patch)
        git(*apply, "-", cwd=source, data=patch)
        if index == 2:
            tree = json.dumps(source_inventory(source), sort_keys=True, separators=(",", ":")).encode()
            if sha256(tree) != verify["verified_tree_sha256"]:
                raise ValueError("reconstructed HW2 tree differs from the retained byte/mode inventory")
            # The independent TP1 harness resolves its HW2 baseline here. Do not
            # silently substitute the TP2 HW3 runtime for that historical tree.
            shutil.copytree(source, stage / "homework-1/tokenspeed")
    return len(source_inventory(source))


def render_report(stage, work, chromium, include_html):
    report = stage / REPORT
    ast = json.loads(run(["pandoc", "--from=gfm-raw_html", "--to=json", str(report)]))
    for link in relative_links(ast):
        target = (report.parent / unquote(urlsplit(link).path)).resolve()
        if link.startswith("//") or not target.is_relative_to(stage) or not target.exists():
            raise ValueError(f"missing or out-of-package report link: {link}")

    css = (ROOT / "scripts/submission.css").read_text(encoding="utf-8") + """
#实验设置与判据, #搬运与-hbm, #单芯片交叉验证与成员贡献 { break-before: page; }
.homework-3 main > table:first-of-type th:first-child { width: 22%; }
.homework-3 main > table:first-of-type th:nth-child(n+2) { width: 39%; }
"""
    title = html.escape("CKPT-3 存储子系统实验报告 — Zhuoxi Zeng — 2501213387")

    def document(tree):
        body = run(["pandoc", "--from=json", "--to=html5", "--wrap=none", "--syntax-highlighting=none"],
                   data=json_bytes(tree)).decode()
        return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
                f'style-src \'unsafe-inline\'; img-src file: data:">'
                f'<title>{title}</title><style>{css}</style></head>'
                f'<body class="{HW}"><main>{body}</main></body></html>')

    if include_html:
        write_file(stage, f"{HW}/report.html", document(ast).encode())
    # Keep relative image paths valid in the print document; never package the print input.
    print_html = report.parent / ".hw3-print.html"
    print_html.write_text(document(pdf_ast(ast)), encoding="utf-8")
    pdf = stage / PDF
    run([chromium, "--headless", "--disable-gpu", "--disable-background-networking",
         "--disable-component-update", "--disable-sync", "--disable-extensions", "--no-first-run",
         "--no-default-browser-check", "--no-proxy-server", "--host-resolver-rules=MAP * ~NOTFOUND",
         "--no-pdf-header-footer", f"--user-data-dir={work / 'browser'}", f"--print-to-pdf={pdf}",
         print_html.as_uri()])
    print_html.unlink()
    pdf.chmod(0o644)
    return verify_pdf(pdf)


def verify_pdf(pdf):
    text = run(["pdftotext", "-layout", str(pdf), "-"]).decode()
    compact = re.sub(r"[\s,]+", "", text)
    required = ("存储子系统实验报告", "2501213387", "陈子睿", "2601112045", "StevenWalxim",
                "2501112130", "0.731", "1.001", "356", "1505")
    missing = [marker for marker in required if marker not in compact]
    if missing or "\ufffd" in text:
        raise ValueError(f"PDF Chinese/identity/result text check failed; missing markers: {missing}")
    info = run(["pdfinfo", str(pdf)]).decode()
    count = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)
    root = ET.fromstring(run(["pdftotext", "-bbox", str(pdf), "-"]))
    pages = root.findall(".//{*}page")
    if not count or not 2 <= len(pages) <= 40 or len(pages) != int(count[1]):
        raise ValueError("unexpected PDF page count")
    for page in pages:
        width, height = (float(page.attrib[key]) for key in ("width", "height"))
        words = page.findall(".//{*}word")
        if sum(len(word.text or "") for word in words) < 40:
            raise ValueError("nearly empty PDF page")
        for word in words:
            x0, y0, x1, y1 = (float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
            if not (0 <= x0 <= x1 <= width + 0.5 and 0 <= y0 <= y1 <= height + 0.5):
                raise ValueError(f"text outside PDF page: {word.text}")
    return len(pages)


def make_zip(stage, output):
    def add(archive, name, data, mode):
        entry = zipfile.ZipInfo(safe_name(name), date_time=(1980, 1, 1, 0, 0, 0))
        entry.create_system = 3
        entry.external_attr = (stat.S_IFREG | mode) << 16
        entry.compress_type = (zipfile.ZIP_STORED if name.endswith((".gz", ".pdf", ".png", ".jpg", ".jpeg"))
                               else zipfile.ZIP_DEFLATED)
        archive.writestr(entry, data)

    hashes = {}
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            name = path.relative_to(stage).as_posix()
            if path.is_symlink() or not path.is_file() or name == SUMS:
                raise ValueError(f"unexpected stage entry: {name}")
            data = path.read_bytes()
            hashes[name] = sha256(data)
            add(archive, name, data, 0o755 if path.stat().st_mode & 0o111 else 0o644)
        add(archive, SUMS, "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode(), 0o644)

    with zipfile.ZipFile(output) as archive:
        rows = [row.split("  ", 1) for row in archive.read(SUMS).decode().splitlines()]
        recorded = {name: digest for digest, name in rows}
        if (recorded != hashes or len(rows) != len(hashes) or len(archive.namelist()) != len(hashes) + 1
                or set(archive.namelist()) != {*recorded, SUMS}):
            raise ValueError("ZIP membership differs from its checksum manifest")
        for name, expected in recorded.items():
            if sha256(archive.read(name)) != expected:
                raise ValueError(f"ZIP checksum differs: {name}")
    return len(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--chromium", default=shutil.which("chromium") or shutil.which("chromium-browser"))
    parser.add_argument("--no-html", action="store_true", help="omit the linked HTML copy from the ZIP")
    args = parser.parse_args()
    for command in ("git", "pandoc", "pdftotext", "pdfinfo"):
        if shutil.which(command) is None:
            parser.error(f"missing local dependency: {command}")
    chromium = shutil.which(args.chromium) if args.chromium else None
    if chromium is None:
        parser.error("headless Chromium not found; pass --chromium")
    chromium = os.path.abspath(chromium)
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Build beside the destinations so replacing each verified artifact is atomic.
    with tempfile.TemporaryDirectory(prefix=".hw3-build-", dir=output) as temp:
        work = Path(temp)
        stage = work / "package"
        stage.mkdir()
        stage_materials(stage)
        source_files = reconstruct(stage, work)
        pages = render_report(stage, work, chromium, not args.no_html)
        members = make_zip(stage, work / ZIP)
        (stage / PDF).replace(output / PDF)
        (work / ZIP).replace(output / ZIP)
    print(f"{output / PDF} ({pages} pages)")
    print(f"{output / ZIP} ({members} checksummed files; {source_files} reconstructed source files)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, ET.ParseError, tarfile.TarError, zipfile.BadZipFile) as error:
        sys.exit(f"HW3 build failed: {error}")
