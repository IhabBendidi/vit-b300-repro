# Measured B300 speed results

One Scaleway B300-SXM-8-288G virtual machine, **8 NVIDIA B300 SXM6 AC GPUs**. Every row below is an actual completed eight-GPU measurement. Per-GPU throughput is the measured node throughput divided by eight; it is not a separate single-GPU benchmark.

The model is ViT-B/16 with 224 × 224 RGB inputs, 12 classes, BF16 autocast and AdamW. Values are medians of three full trials; ranges are the lowest and highest trial rates. Smoke tests are excluded.

## GPU-only training

One batch is generated and copied to each GPU before timing, then reused for forward pass, loss, backpropagation and optimizer updates. Accumulation = 1. Full protocol: 600-second warmup, three 1,200-second trials, followed by a separate 5,000-update run. The table uses only the three timed trials.

| Batch/GPU | Global batch | Images/s, 8 GPUs | Images/s/GPU | Trial range, images/s/node |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 8,192 | 31,730.6 | 3,966.3 | 31,699.8–31,736.5 |
| 2,048 | 16,384 | 32,371.6 | 4,046.5 | 32,369.1–32,376.8 |
| 3,072 | 24,576 | 32,302.7 | 4,037.8 | 32,301.0–32,303.8 |

## Disk-fed training: accumulation = 1

Images are read from the **24 TB local NVMe scratch volume** at `/scratch`, exposed to the guest as `/dev/sdb` (`scratch`). The separate 300 GB `/dev/sda` (`sbs`) boot volume did not serve the timed image reads. The underlying physical SSD model and disk layout were not recorded.

Each input batch is physically read, processed through Zarr, copied to the GPU, and trained. The uncompressed synthetic dataset contains 232,448 images (185,856 training, 46,592 validation), about 140 GB on disk. Full protocol: 120-second warmup plus three 180-second trials for each accumulation setting. Validation is outside the timed trials.

| Batch/GPU | Global batch | Images/s, 8 GPUs | Images/s/GPU | Effective image read GB/s/node |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 8,192 | 6,110.1 | 763.8 | 3.679 |
| 2,048 | 16,384 | 5,302.0 | 662.8 | 3.192 |
| 3,072 | 24,576 | 6,445.4 | 805.7 | 3.881 |

## Disk-fed training: accumulation = 2

Two microbatches per optimizer update; the dataset, GPU count and timed protocol are unchanged.

| Batch/GPU | Global batch | Images/s, 8 GPUs | Images/s/GPU | Effective image read GB/s/node |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 16,384 | 6,025.2 | 753.2 | 3.628 |
| 2,048 | 32,768 | 5,513.3 | 689.2 | 3.320 |
| 3,072 | 49,152 | 6,067.7 | 758.5 | 3.653 |

## What the measurements establish

- At batch 1,024 and accumulation 1, throughput falls from **31,730.6 to 6,110.1 images/s** when using the disk-fed pipeline: 19.3% of the GPU-only rate. These modes have different timing protocols, so this is a mode comparison, not an isolated SSD test.
- The batch-1,024 disk-fed trials span **5,386.3–6,310.8 images/s** and **3.243–3.800 GB/s**. Larger batches did not consistently improve throughput; batch 3,072 with accumulation 1 reached 6,445.4 images/s.
- In the batch-1,024 third trial, one rank recorded **163.1 seconds of queue waiting in 181.0 seconds**, versus approximately 0.01 seconds on two other ranks. Batch preparation and Zarr processing were uneven. These overlapping timing counters are diagnostic clues, not an additive wall-time breakdown or proof of an SSD limit.
- Effective read GB/s is image bytes consumed divided by full training-trial time, using decimal GB. It includes the effect of CPU processing, transfers, training and synchronization; it is not a raw disk bandwidth rating. Physical-read counters were checked against image bytes. No matched write-speed measurement is claimed.
- A batch of 4,096 images/GPU failed the smoke test with CUDA out-of-memory. No throughput is reported for it. The data are procedural images used for a training simulation; these results do not establish biological-model accuracy or convergence.

## Recorded environment

The captured runtime reports PyTorch **2.14.1+cu130**, CUDA **13.0**, NVIDIA driver **580.178.04**, and the default `NCCL_NVLS_ENABLE` setting. Eight GPUs were connected within one node (NV18 reported). The loader selected **2 workers per rank** for the batch-1,024 run; it used one CPU-prefetched batch and overlapped host-to-device transfer.

The original complete package environment and physical SSD model were not retained in this compact summary. Preserve the new environment report when reproducing. Software versions, CPU affinity, filesystem and virtualization may change the result. The summary records the original measurements, not a new run of this extracted repository.

Source provenance and the preserved implementation are described in [methodology](methodology.md) and [source provenance](provenance.json).
