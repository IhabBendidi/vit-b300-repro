# Validation of the extracted code

Checked on 2026-11-02. These are software validation checks, not new B300 speed measurements.

## Completed checks

- **13 tests passed**, covering real CPU forward/backward/optimizer execution and checkpoint output, deterministic image generation, exact full-dataset sizing, Zarr write/read round trips, corrupted-payload rejection, disjoint shard assignment, metadata/hash validation and throughput normalization.
- The documented editable installation and all three command-line entry points were checked. A complete small export was executed with two worker processes through the generation CLI; it produced the store, checksums, report and completion marker.
- Static analysis passed. The Python syntax trees of **21 core model, generator, training and loader definitions** were compared with the source snapshot and were unchanged after extraction and formatting.
- The speed summarizer was checked against the recorded batch-1,024 disk-fed data. The repository's result page contains only the compact measured eight-GPU B300 summary.

Local validation environment: Python 3.12.14, PyTorch 2.15.0, Zarr 3.4.2, NumPy 2.5.4, Pydantic 2.14.2 and psutil 7.2.2 on macOS ARM64. These versions describe the validation environment, not the original GPU runtime. The dependency ranges permit an existing compatible CUDA installation; they do not reconstruct the original complete environment.

## Still requires the target node

Linux physical direct-I/O execution, the full 140 GB export, CUDA/NCCL multi-GPU training and the quoted B300 throughput have **not been rerun as part of this extraction**. On this local machine, the unsupported direct-I/O path was checked to fail explicitly instead of falling back to buffered reads. The Linux branch of that test requires a filesystem supporting `O_DIRECT`.

To reproduce the measured baseline, use the README commands on the intended eight-GPU node and keep its runtime report. A different physical-node or virtual-machine configuration is a new measurement, even when it uses the same GPU model.

## Run the software checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src scripts tests
```
