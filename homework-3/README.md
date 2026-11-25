# Homework 3 — Host offloading on Ascend

CKPT-3 is the November 11–26 assignment on tiered caching. The [original brief](assignment.md) is preserved byte for byte. The [Chinese report](report.md) describes the implementation, recovery boundaries, and evidence. The [local tests](results/local-verification.md) and [hardware results](results/ascend-comparisons.json) cover the scheduler, both models and the tiered-cache experiments.

## Offline acceptance and submission

From the repository root, `python homework-3/scripts/replay_tp2.py` checks the archived patch identity, token and logprob agreement, rank completion events, eviction and final snapshot reclamation. It recomputes all eight comparisons and checks the saved summary. It does not start an NPU job. For the separate TP1 experiments, see Steven’s [report](report-tp1.md) and [runner / evidence guide](README-tp1.md); their progress statements describe that branch at the time of its measurements.

To build the submission from a clean checkout, initialize the pinned submodule, install Pandoc, Chromium, Poppler and CJK fonts, and run `python scripts/build_hw3_submission.py`. The PDF and ZIP appear in `submissions/`. The builder reconstructs both source stages without modifying the checkout, and verifies the embedded checksums. After extracting the ZIP into an empty directory, run `sha256sum --quiet -c SHA256SUMS` and the same replay command from its root. The ZIP already contains the TP2 source under `homework-3/.source` and the independent TP1 baseline under `homework-1/tokenspeed`; do **not** apply the patches again there.

## Prepare the HW3 runtime

The runnable source lives in the ignored `homework-3/.source` directory. The tracked deliverable is the [incremental HW3 patch](patches/ascend-host-cache.patch). To reproduce the stage on a new machine, initialize the pinned TokenSpeed submodule and run from the repository root.

```bash
ROOT="$PWD"
STAGE="$ROOT/homework-3/.source"
mkdir "$STAGE"
git -C homework-1/tokenspeed archive 52bdb9763bb2da3d61336398072671885cc0a33e |
  tar -xf - -C "$STAGE"
stage_git() {
  GIT_CEILING_DIRECTORIES="$ROOT/homework-3" git -C "$STAGE" "$@"
}
stage_git apply --whitespace=error "$ROOT/homework-1/patches/ascend-qwen3.5.patch"
stage_git apply --whitespace=error "$ROOT/homework-2/patches/hccl-torch29.patch"
stage_git apply --check --whitespace=error "$ROOT/homework-3/patches/ascend-host-cache.patch"
stage_git apply --whitespace=error "$ROOT/homework-3/patches/ascend-host-cache.patch"
```

Build the scheduler in a private HW3 inference environment that can import the course's existing Ascend runtime. Set `PY` to that environment's Python. The trace needs the updated scheduler bindings.

```bash
: "${PY:?set to the Python executable in the HW3 inference environment}"
"$PY" -m pip install --force-reinstall --no-deps ./homework-3/.source/tokenspeed-scheduler
```

On the course server we used system CMake and two staged wheels, `nanobind==3.1.0` and `tokenspeed-spdlog==1.15.1`. The following equivalent offline build puts the extension in a private directory. Keep that directory on `PYTHONPATH` when running the matrix.

```bash
"$PY" -m pip install --no-index --no-deps --target "$PWD/builddeps" wheels/*.whl
PYTHONPATH="$PWD/builddeps" cmake -S "$STAGE/tokenspeed-scheduler" -B native \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PY" \
  -DOPENSSL_ROOT_DIR=/opt/miniconda3 -DCMAKE_BUILD_TYPE=Release
cmake --build native -j6
mkdir -p deps/tokenspeed_scheduler
cp "$STAGE/tokenspeed-scheduler/python/tokenspeed_scheduler/__init__.py" \
  native/tokenspeed_scheduler_ext*.so deps/tokenspeed_scheduler/
```

## Run the Ascend matrix

Use the [HW2 course-server environment setup](../homework-2/README.md#server-reproduction-templates). Check `npu-smi info` before choosing a permitted same-card pair. Run the following from the repository root. Choose a new output directory and an available rendezvous port. The launcher captures its own processes' HBM snapshots and compares the effective cache-block budgets across runs.

```bash
: "${PY:?set to the Python executable in the HW3 inference environment}"
: "${PAIR:?set to a permitted same-card pair listed by npu-smi}"
: "${OUT:?set to a new results directory}"
: "${PORT:?set to an available job-local rendezvous port}"
mkdir "$OUT"
STAGE="$PWD/homework-3/.source"
export PYTHONPATH="$PWD/deps:$STAGE/python:$STAGE/tokenspeed-kernel/python:$STAGE/tokenspeed-kernel-npu/python"
RUN=homework-3/scripts/run_offload.py

trial() {
  name=$1
  shift
  mkdir -p "$OUT/$name"
  timeout --signal=TERM --kill-after=20s 900s "$PY" "$RUN" run \
    --tp 2 --physical-devices "$PAIR" --port "$PORT" --rounds 4 --prompt-length 192 --output-tokens 8 \
    --model "$MODEL" --memory-fraction "$FRACTION" --dtype "$DTYPE" \
    "${PRECISION[@]}" "$@" --output-dir "$OUT/$name/run" > "$OUT/$name/worker.log" 2>&1
  rc=$?
  if [ -d "$OUT/$name/run" ]; then
    mv "$OUT/$name/worker.log" "$OUT/$name/run/worker.log"
  fi
  return "$rc"
}

compare() {
  baseline=$1 trial_name=$2
  shift 2
  "$PY" "$RUN" compare "$OUT/$baseline/run" "$OUT/$trial_name/run" \
    --output "$OUT/$trial_name/comparison.json" "$@"
}

matrix() {
  tag=$1
  trial "$tag-high-off" --device-tokens 9216 --pattern high --host-ratio 0 &&
  trial "$tag-high-large" --device-tokens 9216 --pattern high --host-ratio 4 &&
  trial "$tag-high-small" --device-tokens 9216 --pattern high --host-ratio 0.5 &&
  trial "$tag-low-off" --device-tokens 9216 --pattern low --host-ratio 0 &&
  trial "$tag-low-on" --device-tokens 9216 --pattern low --host-ratio 4 &&
  trial "$tag-retract-off" --device-tokens 16384 --pattern retract --host-ratio 0 &&
  trial "$tag-retract-on" --device-tokens 16384 --pattern retract --host-ratio 0 \
    --snapshot-gb 0.25 --force-interval 4 || return
  compare "$tag-high-off" "$tag-high-large" --require-copy WriteBackDone --require-copy LoadBackDone &&
  compare "$tag-high-off" "$tag-high-small" --require-eviction &&
  compare "$tag-low-off" "$tag-low-on" --require-copy WriteBackDone &&
  compare "$tag-retract-off" "$tag-retract-on" --require-copy SnapshotDone --require-copy RestoreDone
}

MODEL=homework-1/models/Qwen3.5-0.8B
FRACTION=0.12
DTYPE=bfloat16
PRECISION=()
matrix 0.8B

MODEL=homework-2/models/Qwen3.5-4B
FRACTION=0.25
DTYPE=float16
PRECISION=(--lm-head-compute-dtype float32)
matrix 4B
```

The recorded runs used Ascend 910 physical chips 6,7 for 0.8B and 4,5 for 4B. Choose an available pair and a distinct HCCL port range for each concurrent job. The `9216` and `16384` token caps yielded 9 and 16 usable Device LCM blocks per rank. Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The reference and trial for each model use the same pair and precision.

To inspect the recorded experiment offline, extract the [raw hardware runs](results/ascend-runs.tar.gz) into an empty directory. Each directory contains its `result.json`, `worker.log` and HBM snapshots. The same `compare` command accepts those directories, with no model, accelerator or external Python package required. `post_warmup_output_tokens_per_second` includes every generated token and request wall time after the first seed and decode shapes have run. The separate full-trace throughput includes cold kernel startup.

The trace records allocated and claimed Host bytes, Device LCM block occupancy and reclaimable capacity, Host hits, misses and capacity evictions, and completed copy bytes and latency by rank. A copy's reported duration spans submission through completion observation. It includes stream waits, staging, and polling. End-to-end batch latency also includes prefill of the uncached suffix. The comparison records both quantities and the output token throughput.
