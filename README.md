# ViT-B/16 on 8 B300 GPUs

Reproduce a ViT compute benchmark and the same model trained from images stored in local Zarr files. Start with the **[measured B300 speed summary](docs/b300-results.md)**.

The repository contains the model, procedural image generator, GPU-only training, disk-fed training and a compact summary of completed eight-GPU B300 measurements.

## Run on a B300 node

Use Linux, Python 3.12, eight visible CUDA GPUs, a B300-compatible CUDA PyTorch installation, and at least **151 GB free** on the selected scratch filesystem. The standard generation command writes about **140 GB**. Run these benchmarks exclusively on the node so competing workloads do not distort the result.

### 1. Install and inspect the environment

Preserve the node's compatible CUDA PyTorch installation. The recorded baseline used PyTorch `2.14.1+cu130`, CUDA 13.0 and driver 580.178.04. The package does not choose a CUDA wheel for you; [PyTorch's installer](https://pytorch.org/get-started/locally/) supplies the appropriate build when one is not already installed.

```bash
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_environment.py --require-8-gpus
```

The environment report prints to the terminal. Save it with each run; the training outputs also record CUDA, PyTorch, driver and GPU details. `docs/validation.md` distinguishes local software checks from GPU validation.

### 2. Measure GPU-only training

```bash
torchrun --standalone --nproc-per-node=8 -m vit_benchmark.imaging \
  --config config/b300-8gpu-b1024.toml --profile smoke
```

After the smoke run succeeds, repeat with `--profile full`. Full timing is 10 minutes of warmup + three 20-minute trials + 5,000 updates and untimed validation/setup. At the recorded rate, the timed fixed-update portion alone is about 22 minutes. The output is `runs/gpu-only-b1024/gpu-only-b1024.json`; a later run with the same configuration replaces that output, so move completed run folders before repeating.

### 3. Generate the images directly on the target disk

Confirm that `/scratch` is the intended data disk using `findmnt -T /scratch`. For another mount, replace both paths below.

```bash
python -m vit_benchmark.imaging_zarr \
  --config config/b300-8gpu-b1024.toml \
  --scratch-root /scratch \
  --output /scratch/vit-benchmark/images.zarr \
  --report runs/dataset-generation.json \
  --target-bytes 140000000000 --max-bytes 150000000000 --workers 32 \
  --allow-same-filesystem
python scripts/verify_dataset.py /scratch/vit-benchmark/images.zarr
```

`--allow-same-filesystem` permits the checkout's result folder and dataset to share the chosen filesystem; it does not select or mount a disk. Generation refuses to overwrite an existing dataset or incomplete export. It creates Zarr metadata, checksums, a manifest and a completion marker. The verification command reads all files to check SHA-256 hashes; this is outside training timing. Generation duration depends on CPU and write performance.

### 4. Measure disk-fed training

```bash
torchrun --standalone --nproc-per-node=8 -m vit_benchmark.zarr_training \
  --config config/b300-8gpu-b1024.toml \
  --dataset /scratch/vit-benchmark/images.zarr \
  --batch-size 1024 --profile smoke --output runs/zarr-b1024-smoke
```

After smoke succeeds, run the same command with `--profile full --output runs/zarr-b1024-full`. The full run measures accumulation 1 and 2: 22 minutes of timed phases, plus setup, loader tuning and validation. Workers are tuned from 1, 2 and 4 per rank; use `--workers 2` to fix the batch-1,024 baseline's selected value, or `--trace` for an additional untimed profiler trace.

Images are read with Linux `O_DIRECT`. The program fails if direct reads cannot be verified; it never silently substitutes memory-cached reads. This intentionally stresses physical input delivery, even when the small dataset could fit in system RAM.

### 5. Read the new speed summary

```bash
python scripts/summarize.py runs/gpu-only-b1024/gpu-only-b1024.json
python scripts/summarize.py runs/zarr-b1024-full/summary.csv
```

The summaries report images/s for the node and per GPU, trial ranges and, for disk-fed training, effective image read GB/s. Detailed new diagnostics stay under the ignored `runs/` folder.

## Batch-size experiments

Configs for **1,024, 2,048 and 3,072 images/GPU** are included. For GPU-only training select the matching config. For disk-fed training also set the matching `--batch-size` and a new `--output` directory. Do not compare different global batches as evidence of equal learning progress.

## Files to read

- [Measured B300 speeds](docs/b300-results.md): the compact result tables.
- [Methodology](docs/methodology.md): architecture, synthetic data, timing and diagnostic interpretation.
- [Validation](docs/validation.md): checks performed on this port and remaining GPU checks.
- [License](LICENSE): MIT, with the original copyright notice retained.
