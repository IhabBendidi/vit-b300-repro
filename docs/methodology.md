# Methodology and reproduction boundaries

## Architecture and optimization

ViT-B/16 uses 224 × 224 × 3 images, a stride-16 patch convolution, 196 patches plus a class token, learned positional embeddings, 12 pre-normalized transformer blocks, width 768, 12 attention heads and an MLP width of 3,072. Each block uses `nn.MultiheadAttention` with `need_weights=False`, GELU and LayerNorm epsilon 1e-6. The final class token feeds a 12-class head. There is no added augmentation, compilation or alternative attention implementation.

Training uses cross-entropy, BF16 autocast with FP32 model parameters, and AdamW (`lr=0.001`, `weight_decay=0.01`). Eight processes use PyTorch DistributedDataParallel. The GPU-only path uses the default distributed reduction; the disk-fed path retains the measured all-reduce instrumentation hook. Each baseline has 1,024 images per GPU and accumulation 1, giving 8,192 images per update. The disk-fed runner additionally measures accumulation 2.

## Procedural data

The generator constructs microscopy-like stripes, blobs and rings, with seeded noise and a compound-linked label. It is deterministic within a fixed software/runtime environment. It is synthetic, not a biological dataset. Training compounds are 0–79 and validation compounds 80–99. The seed is 20261005. Regeneration on another library version or platform is not guaranteed to produce byte-identical floating-point payloads.

GPU-only training builds a logical 20,000-sample dataset with a 4,000-sample validation split. Before each timed phase, each rank preloads a batch onto its GPU, then reuses that batch throughout the phase. This measures repeated forward/backward/optimizer throughput, not reading new images or convergence.

The disk-fed store targets 140,000,000,000 decimal bytes with a 150,000,000,000-byte cap. Shard alignment yields 185,856 training and 46,592 validation images. Images are float32 NCHW; labels are int64 and compound IDs uint16. Zarr v3 image chunks contain 8 images and image shards 256 images. A shard contains about 154.14 MB. Data are uncompressed, with CRC32C integrity checks. The training image payload is 602,112 bytes/image.

Generation writes directly into a new `.incomplete` directory on the selected disk. It checks samples against regeneration, creates file hashes and metadata receipts, then renames the completed directory. `scripts/verify_dataset.py` verifies all payload hashes before training. The training runner checks metadata identity and Zarr validates chunk checksums when decoding.

## Reading and training

The local loader reads complete image shard files using aligned `O_DIRECT` reads and verifies Linux per-thread physical-read counters. In a VM, these counters confirm I/O through the guest block layer; they do not independently reveal host-side caching. Metadata and labels use buffered reads. Shards are decoded using the official Zarr codec path, including integrity checks, then collated into pinned CPU memory. Each rank owns distinct shard groups; deterministic affine permutations change their order between epochs. Only complete global batch groups are used; omitted shards rotate between epochs, and larger batches change the omitted count.

Reader counts 1, 2 and 4 are timed before the run and the best setting is selected. There is one batch under construction per rank, a bounded CPU prefetch and an asynchronous host-to-device stream. The full disk run retains one model/optimizer across warmup and three trials for each accumulation setting. The GPU-only runner recreates model/optimizer for each timed phase, as in the original benchmark. These differences are retained to reproduce the measurements, and the two modes should not be interpreted as a controlled raw-disk-only comparison.

## Timing and counters

| Item | Meaning |
| --- | --- |
| Images/s/node | Sum of processed samples across ranks divided by the slowest rank's measured elapsed time |
| Images/s/GPU | Node throughput divided by eight for the recorded eight-GPU runs |
| Effective read GB/s | Image bytes consumed divided by full training-trial time; decimal GB |
| Physical read bytes | Linux read accounting checked against the image payload; not a filesystem cache hit count |
| Queue/build/decode timing | Diagnostic counters accumulated within the loader; work may overlap across threads, CPU and GPU |

Do not add overlapping read/decode/copy/compute counters to reconstruct wall time. In particular, summed decoder time across worker threads can exceed trial elapsed time. A low effective GB/s does not by itself prove the storage device has that raw bandwidth limit. A GPU utilization percentage is not the percentage of peak useful training work.

The baseline uses physical reads to isolate an input-delivery scenario. A production RAM cache, different dtype/compression, alternative codec path, deeper prefetch or changed CPU affinity may improve speed, but changes must be reported as separate experiments.

## References

- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [Zarr array and sharding documentation](https://zarr.readthedocs.io/en/stable/user-guide/arrays/)
- [Scaleway scratch-storage documentation](https://www.scaleway.com/en/docs/gpu/how-to/use-scratch-storage-h100-instances/)

The original MIT notice is retained. Source-file hashes and the extraction changes are in `provenance.json`. No original dataset files or full measurement artifacts are distributed here.
