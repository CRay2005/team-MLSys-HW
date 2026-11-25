# Local verification

The final patch applied cleanly to a temporary checkout of TokenSpeed `52bdb9763bb2da3d61336398072671885cc0a33e` after applying the repository's HW1 and HW2 patches. All 11 HW3 patch paths matched the development source byte for byte after reconstruction. The reconstructed HW3 runtime was retained under `homework-3/.source` so the HW2 runtime can still be used directly. The assignment file also matched the original download byte for byte.

| Check | Result | Coverage |
| --- | --- | --- |
| C++ scheduler cache scenarios, cache coordinator, and state checkpoints | 297 passed | Host hits after Device eviction, Host capacity eviction, ACK ordering, state checkpoint recovery, and block reclamation |
| Scheduler Python suite | 279 passed | Rebuilt native Python bindings and scheduler lifecycle |
| Host storage, layerwise load, and transfer descriptor suites | 66 passed, 5 skipped | Includes the CPU byte-exact round trip using production descriptor expansion and byte copies. Hardware cases require Ascend or CUDA. |
| HW3 runner tests | 2 passed | Workload parity and TokenSpeed's per-rank logger format |
| HW3 Python lint and reproduction shell syntax | Passed | Runner, tests, and documented shell commands |
| HW2 offline suite | 196 passed | Includes byte-for-byte reconstruction of the HW2 runtime |

The test binary used the same scheduler core as the rebuilt Python wheel. The local machine has CPU PyTorch and no Ascend devices. The subsequent course-server runs and comparisons are recorded in the [hardware report](../report.md).
