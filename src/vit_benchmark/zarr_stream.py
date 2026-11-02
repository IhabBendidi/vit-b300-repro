"""Bounded local Zarr readers; image reads require physical O_DIRECT I/O."""

from __future__ import annotations

import json
import math
import mmap
import os
import random
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

SCRATCH = Path("/scratch/vit-benchmark/images.zarr")


def emit(event: str, **values: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, "timestamp": time.time(), **values}) + "\n")
    sys.stdout.flush()


def thread_read_bytes() -> int:
    return int(
        dict(line.split(":", 1) for line in Path("/proc/thread-self/io").read_text().splitlines())[
            "read_bytes"
        ]
    )


def direct_read(path: Path) -> tuple[memoryview, dict[str, float]]:
    """Aligned direct I/O with mandatory physical-read evidence, no fallback.

    Anonymous MAP_SHARED memory is an aligned transfer buffer, not a file map.
    EOF may return an unaligned short final read; the requested size is aligned.
    """
    if not sys.platform.startswith("linux") or not hasattr(os, "O_DIRECT"):
        raise RuntimeError("Physical direct I/O requires Linux; buffered fallback is forbidden")
    before = thread_read_bytes()
    started = time.perf_counter()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
    try:
        size = os.fstat(fd).st_size
        capacity = (size + 4095) // 4096 * 4096
        buffer = mmap.mmap(-1, capacity)
        try:
            received = os.preadv(fd, [buffer], 0)
            if received != size:
                raise OSError(f"Short direct read: {received}/{size}: {path}")
            physical = thread_read_bytes() - before
            if physical < size:
                raise RuntimeError(f"Physical read verification failed: {physical}/{size}: {path}")
            read_seconds = time.perf_counter() - started
            blob = memoryview(buffer)[:size]
        except BaseException:
            buffer.close()
            raise
    finally:
        os.close(fd)
    return blob, {
        "read_seconds": read_seconds,
        "bytes": size,
        "physical_bytes": physical,
        "requests": 0,
        "retries": 0,
        "first_byte_seconds": 0,
        "buffer_copy_seconds": 0.0,
    }


def decode_shard(metadata: bytes, blob: bytes | memoryview, shard: int) -> np.ndarray:
    """Use the official Zarr codec stack, including its CRC32C validation."""
    meta = json.loads(metadata)
    ndim = len(meta["shape"])
    key = "c/" + "/".join([str(shard), *(["0"] * (ndim - 1))])
    factory = default_buffer_prototype().buffer.from_bytes
    store = MemoryStore({"zarr.json": factory(metadata), key: factory(blob)}, read_only=True)
    array = zarr.open_array(store=store, mode="r")
    width = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    return np.asarray(array[shard * width : (shard + 1) * width])


def shard_order(count: int, rank: int, world: int, epoch: int, seed: int) -> list[int]:
    # An affine permutation avoids an O(dataset_size) shuffle table. This helper
    # returns just this rank's current epoch for the small pilot; iterator below
    # uses the same permutation without a list even for petabyte-scale datasets.
    return list(iter_shards(count, rank, world, epoch, seed))


def iter_shards(
    count: int, rank: int, world: int, epoch: int, seed: int, shards_per_batch: int = 4
) -> Iterator[int]:
    if shards_per_batch < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid shard assignment parameters")
    rng = random.Random(seed + epoch)
    multiplier = rng.randrange(1, count)
    while math.gcd(multiplier, count) != 1:
        multiplier = (multiplier + 1) % count or 1
    offset = rng.randrange(count)
    usable = count // (world * shards_per_batch) * (world * shards_per_batch)
    if usable == 0:
        raise ValueError("Dataset must supply one complete batch of shards per rank")
    for position in range(rank * shards_per_batch, usable, world * shards_per_batch):
        for j in range(shards_per_batch):
            yield ((position + j) * multiplier + offset) % count


class ShardDataset:
    def __init__(self, root: Path, split: str = "train"):
        self.root, self.split = root, split
        zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
        self.metadata = {}
        for name in ("images", "labels"):
            key = f"{split}/{name}/zarr.json"
            self.metadata[name] = self.read(key, image=False)[0]
        meta = json.loads(self.metadata["images"])
        self.count = meta["shape"][0]
        self.width = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        if (
            self.width != 256
            or meta["shape"][1:] != [3, 224, 224]
            or meta["data_type"] != "float32"
        ):
            raise ValueError("Expected the immutable pilot dataset: 256-image float32 NCHW shards")
        self.shards = self.count // self.width

    def read(self, key: str, image: bool) -> tuple[Any, dict[str, float]]:
        if image:
            return direct_read(self.root / key)
        started = time.perf_counter()
        blob = (self.root / key).read_bytes()
        return blob, {
            "read_seconds": time.perf_counter() - started,
            "bytes": len(blob),
            "physical_bytes": 0,
            "requests": 0,
            "retries": 0,
        }

    def load(self, shard: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        blob, stats = self.read(f"{self.split}/images/c/{shard}/0/0/0", image=True)
        stats["image_bytes"] = len(blob)
        started = time.perf_counter()
        images = decode_shard(self.metadata["images"], blob, shard)
        stats["decode_seconds"] = time.perf_counter() - started
        del blob
        labels_blob, label_stats = self.read(f"{self.split}/labels/c/{shard}", image=False)
        started = time.perf_counter()
        labels = decode_shard(self.metadata["labels"], labels_blob, shard)
        stats["decode_seconds"] += time.perf_counter() - started
        for key, value in label_stats.items():
            stats[key] = stats.get(key, 0) + value
        return images, labels, stats


class BatchLoader:
    """Exactly one batch under construction; one future limits CPU prefetch.

    The consumer owns at most its current batch plus this single next batch.
    Persistent threads share neither image caches nor growing epoch buffers.
    """

    def __init__(
        self,
        dataset: ShardDataset,
        rank: int,
        world: int,
        workers: int,
        seed: int,
        batch_size: int = 1024,
    ):
        if batch_size < dataset.width or batch_size % dataset.width:
            raise ValueError("Batch size must be a positive multiple of the image shard size")
        self.dataset, self.rank, self.world, self.seed = dataset, rank, world, seed
        self.batch_size = batch_size
        self.shards_per_batch = batch_size // dataset.width
        if dataset.shards < world * self.shards_per_batch:
            raise ValueError("Dataset cannot supply one full batch per rank")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.producer = ThreadPoolExecutor(max_workers=1)
        self.epoch = 0
        self.order = iter(iter_shards(dataset.shards, rank, world, 0, seed, self.shards_per_batch))
        self.pending: Future[tuple[torch.Tensor, torch.Tensor, dict[str, float]]] | None = None

    def build(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        started = time.perf_counter()
        indices = []
        for _ in range(self.shards_per_batch):
            try:
                indices.append(next(self.order))
            except StopIteration:
                self.epoch += 1
                self.order = iter(
                    iter_shards(
                        self.dataset.shards,
                        self.rank,
                        self.world,
                        self.epoch,
                        self.seed,
                        self.shards_per_batch,
                    )
                )
                indices.append(next(self.order))
        pin_start = time.perf_counter()
        images = torch.empty((self.batch_size, 3, 224, 224), dtype=torch.float32, pin_memory=True)
        labels = torch.empty(self.batch_size, dtype=torch.int64, pin_memory=True)
        stats = {"pin_allocate_seconds": time.perf_counter() - pin_start, "collate_seconds": 0.0}
        futures: list[Any] = [self.pool.submit(self.dataset.load, index) for index in indices]
        for j, future in enumerate(futures):
            image, label, read_stats = future.result()
            copy_start = time.perf_counter()
            start, stop = j * self.dataset.width, (j + 1) * self.dataset.width
            images[start:stop].copy_(torch.from_numpy(image))
            labels[start:stop].copy_(torch.from_numpy(label))
            stats["collate_seconds"] += time.perf_counter() - copy_start
            for key, value in read_stats.items():
                stats[key] = stats.get(key, 0) + value
            futures[j] = None  # Release each decoded shard immediately after collation.
        stats["batch_build_seconds"] = time.perf_counter() - started
        return images, labels, stats

    def next(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        started = time.perf_counter()
        if self.pending is None:
            self.pending = self.producer.submit(self.build)
        batch = self.pending.result()
        self.pending = self.producer.submit(self.build)
        batch[2]["queue_wait_seconds"] = time.perf_counter() - started
        return batch

    def close(self) -> None:
        self.producer.shutdown(wait=True, cancel_futures=True)
        self.pending = None
        self.pool.shutdown(wait=True, cancel_futures=True)
