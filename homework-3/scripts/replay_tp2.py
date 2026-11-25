"""Replay the submitted CKPT-3 matrix without weights, network or an accelerator."""
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace

from run_offload import compare, require, workload

HW = Path(__file__).resolve().parents[1]
CASES = (
    ('high-off', 'high-large', ('WriteBackDone', 'LoadBackDone')),
    ('high-off', 'high-small', ()),
    ('low-off', 'low-on', ('WriteBackDone',)),
    ('retract-off', 'retract-on', ('SnapshotDone', 'RestoreDone')),
)


def main():
    stored = json.loads((HW / 'results/ascend-comparisons.json').read_text())
    patch = hashlib.sha256((HW / 'patches/ascend-host-cache.patch').read_bytes()).hexdigest()
    models = ('0.8B', '4B')
    cases = {f'{m}-{s}' for m in models for base, trial, _ in CASES for s in (base, trial)}
    names = {Path(case, file) for case in cases for file in
             ('result.json', 'worker.log', 'hbm-before.txt', 'hbm-loaded.txt', 'hbm-after.txt')}
    expected_keys = {f'{m}-{trial}' for m in models for _, trial, _ in CASES}
    require(set(stored) == expected_keys, 'unexpected comparison matrix')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(HW / 'results/ascend-runs.tar.gz', 'r:gz') as archive:
            for member in archive:
                if member.isdir():
                    continue
                name = Path(member.name)
                require(name in names and member.isfile() and member.size <= 32 * 1024**2,
                        f'unexpected run artifact: {name}')
                names.remove(name)
                dest = root / name
                dest.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as src:
                    dest.write_bytes(src.read())
        require(not names, 'missing run artifacts')
        counts = {}
        for case in sorted(cases):
            doc = json.loads((root / case / 'result.json').read_bytes())
            require(doc['patch_sha256'] == patch and doc['tp'] == 2, f'{case}: source / TP mismatch')
            model, pattern, mode = case.split('-')
            require(model in doc['config']['model'] and doc['pattern'] == pattern, f'{case}: model / pattern mismatch')
            cfg = doc['config']
            if mode == 'off':
                require(cfg['disable_kvstore'] and not cfg['retraction_snapshot_host_gb'] and not cfg['retraction_snapshot_ratio'], f'{case}: baseline has Host caching enabled')
            token_count = 0
            for batch, (prompts, n) in zip(doc['runs'], workload(pattern, 192, 4, 8), strict=True):
                for row, prompt in zip(batch['rows'], prompts, strict=True):
                    require(row['prompt'] == prompt and len(row['tokens']) == len(row['logprobs']) == n
                            and all(map(math.isfinite, row['logprobs'])), f'{case}: invalid output or workload')
                    token_count += n
            counts[case] = token_count
        compared = 0
        for model in models:
            for base, trial, copies in CASES:
                key = f'{model}-{trial}'
                out = root / 'comparison.json'
                result = compare(SimpleNamespace(baseline=root / f'{model}-{base}', trial=root / key,
                                                 output=out, atol=0.15, require_copy=copies,
                                                 require_eviction=trial == 'high-small'))
                require(result == 0 and json.loads(out.read_bytes()) == stored[key], f'{key}: comparison differs')
                compared += counts[key]
        require(compared == 356, 'unexpected token count')
        print(f'{len(cases)} Ascend TP2 runs, {len(stored)} reproduced comparisons, {compared} matched output tokens.')


if __name__ == '__main__':
    main()
