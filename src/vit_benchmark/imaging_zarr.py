"""Materialize procedural imaging data into a bounded, sharded local Zarr store."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import shutil
import socket
import stat
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from zarr.codecs import Crc32cCodec

from vit_benchmark.config import load_config
from vit_benchmark.imaging import SyntheticImageSpec, SyntheticImagingDataset


@dataclass(frozen=True)
class ExportPlan:
    image_size: int
    train_count: int
    validation_count: int
    chunk_images: int
    shard_images: int
    logical_bytes: int
    max_bytes: int


def make_plan(
    image_size: int,
    target_bytes: int,
    max_bytes: int,
    chunk_images: int = 8,
    shard_images: int = 256,
) -> ExportPlan:
    if image_size < 1 or chunk_images < 1 or shard_images < 1 or shard_images % chunk_images:
        raise ValueError("Image size must be positive and shards must contain whole chunks")
    if target_bytes <= 0 or target_bytes * 1.01 + 1_000_000_000 > max_bytes:
        raise ValueError("Target must leave at least 1% plus 1 GB below the storage cap")
    # float32 NCHW images, int64 labels and uint16 compounds; sample IDs are row indices.
    bytes_per_image = 3 * image_size * image_size * 4 + 8 + 2
    shards = target_bytes // (bytes_per_image * shard_images)
    if shards < 2:
        raise ValueError("Target must fit at least one shard in each split")
    validation_shards = max(1, round(shards * 0.2))
    return ExportPlan(
        image_size,
        (shards - validation_shards) * shard_images,
        validation_shards * shard_images,
        chunk_images,
        shard_images,
        shards * shard_images * bytes_per_image,
        max_bytes,
    )


def log(event: str, **values: Any) -> None:
    print(
        json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **values}),
        flush=True,
    )


def configure_worker() -> None:
    torch.set_num_threads(1)
    zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})


def write_shard(task: tuple[str, str, int, int, int, int, int, tuple[int, ...]]) -> dict[str, Any]:
    output, split, start, count, size, seed, total, compounds = task
    dataset = SyntheticImagingDataset(SyntheticImageSpec(seed, total, size, 12, compounds))
    images = np.empty((count, 3, size, size), dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    compound_ids = np.empty(count, dtype=np.uint16)
    for offset in range(count):
        sample = start + offset
        image, label = dataset[sample]
        images[offset] = image.numpy()
        labels[offset] = int(label)
        compound_ids[offset] = compounds[sample % len(compounds)]
    group: Any = zarr.open_group(output, mode="r+")[split]
    # Each task owns a whole shard, including the aligned label/compound arrays.
    # No workers modify shared metadata or the same storage object.
    group["images"][start : start + count] = images
    group["labels"][start : start + count] = labels
    group["compound_ids"][start : start + count] = compound_ids
    return {"split": split, "start": start, "count": count}


def directory_bytes(path: Path) -> tuple[int, int]:
    logical = allocated = 0
    for item in path.rglob("*"):
        try:
            info = item.stat()
        except FileNotFoundError:
            # Zarr atomically renames temporary objects while workers are writing.
            continue
        allocated += info.st_blocks * 512
        if stat.S_ISREG(info.st_mode):
            logical += info.st_size
    return logical, allocated


def create_store(path: Path, plan: ExportPlan, attrs: dict[str, Any]) -> None:
    root = zarr.open_group(str(path), mode="w-", zarr_format=3)
    root.attrs.update(attrs)
    for split, count in (("train", plan.train_count), ("validation", plan.validation_count)):
        group = root.create_group(split)
        group.create_array(
            "images",
            shape=(count, 3, plan.image_size, plan.image_size),
            dtype="float32",
            chunks=(plan.chunk_images, 3, plan.image_size, plan.image_size),
            shards=(plan.shard_images, 3, plan.image_size, plan.image_size),
            compressors=[Crc32cCodec()],
            dimension_names=("sample", "channel", "height", "width"),
        )
        for name, dtype in (("labels", "int64"), ("compound_ids", "uint16")):
            group.create_array(
                name,
                shape=(count,),
                dtype=dtype,
                chunks=(plan.shard_images,),
                compressors=[Crc32cCodec()],
                dimension_names=("sample",),
            )


def verify_samples(path: Path, plan: ExportPlan, seed: int) -> int:
    root: Any = zarr.open_group(str(path), mode="r")
    checked = 0
    for split, count, compounds in (
        ("train", plan.train_count, tuple(range(80))),
        ("validation", plan.validation_count, tuple(range(80, 100))),
    ):
        dataset = SyntheticImagingDataset(
            SyntheticImageSpec(seed, count, plan.image_size, 12, compounds)
        )
        positions = {0, count - 1, plan.shard_images - 1, plan.shard_images}
        positions.update(int(x) for x in np.random.default_rng(seed).integers(0, count, size=12))
        for sample in sorted(x for x in positions if x < count):
            image, label = dataset[sample]
            np.testing.assert_array_equal(root[split]["images"][sample], image.numpy())
            assert int(root[split]["labels"][sample]) == int(label)
            assert int(root[split]["compound_ids"][sample]) == compounds[sample % len(compounds)]
            checked += 1
    return checked


def checksum_store(path: Path) -> int:
    paths = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "checksums.sha256", "_SUCCESS"}
    )
    with (path / "checksums.sha256").open("w") as output:
        for index, item in enumerate(paths):
            with item.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            output.write(f"{digest}  {item.relative_to(path)}\n")
            if index % 100 == 0:
                log("checksum_progress", files_complete=index + 1, files_total=len(paths))
    return len(paths)


def export(args: argparse.Namespace) -> dict[str, Any]:
    configure_worker()
    config = load_config(args.config)
    settings = config.imaging
    if (
        settings is None
        or settings.class_count != 12
        or settings.compound_count != 100
        or settings.train_compound_count != 80
    ):
        raise ValueError(
            "Expected the existing 12-class imaging generator and 80/20 compound split"
        )
    plan = make_plan(
        settings.image_size, args.target_bytes, args.max_bytes, args.chunk_images, args.shard_images
    )
    destination = args.output.resolve()
    scratch = Path(args.scratch_root).resolve()
    if not destination.is_relative_to(scratch) or destination == scratch:
        raise ValueError("Dataset output must be inside the designated scratch filesystem")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.run.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        not args.allow_same_filesystem
        and os.stat(destination.parent).st_dev == os.stat(config.run.output_dir.resolve()).st_dev
    ):
        raise ValueError("Scratch output must not share the results/home filesystem")
    partial = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or partial.exists():
        raise FileExistsError("Refusing to overwrite an existing dataset or partial export")
    if shutil.disk_usage(destination.parent).free < args.max_bytes + 1_000_000_000:
        raise RuntimeError("Insufficient scratch capacity for the bounded export")
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "generating",
        "format": "zarr_v3",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dataset_path": str(destination),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generator": "synthetic_imaging_v1",
        "seed": config.run.seed,
        "generator_source_sha256": hashlib.sha256(
            Path(inspect.getfile(SyntheticImagingDataset)).read_bytes()
        ).hexdigest(),
        "class_count": 12,
        "train_compounds": list(range(80)),
        "validation_compounds": list(range(80, 100)),
        "layout": "NCHW",
        "dtype": "float32",
        "compression": "none",
        "chunk_checksum": "crc32c",
        "size_units": "decimal bytes (1 GB = 1,000,000,000 bytes)",
        "plan": asdict(plan),
        "config_source": str(Path(args.config).resolve()),
        "zarr_version": zarr.__version__,
        "torch_version": torch.__version__,
        "workers": args.workers,
        "split_sample_ids": "zero-based row index within each split",
    }
    create_store(partial, plan, manifest)
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    tasks = [
        (
            str(partial),
            split,
            start,
            plan.shard_images,
            plan.image_size,
            config.run.seed,
            count,
            compounds,
        )
        for split, count, compounds in (
            ("train", plan.train_count, tuple(range(80))),
            ("validation", plan.validation_count, tuple(range(80, 100))),
        )
        for start in range(0, count, plan.shard_images)
    ]
    log("export_start", **manifest, total_shards=len(tasks))
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=configure_worker,
    ) as pool:
        futures = [pool.submit(write_shard, task) for task in tasks]
        for future in as_completed(futures):
            shard = future.result()
            completed += 1
            if completed == 1 or completed % 8 == 0 or completed == len(tasks):
                logical, allocated = directory_bytes(partial)
                if max(logical, allocated) > plan.max_bytes:
                    raise RuntimeError("Dataset storage cap exceeded")
                elapsed = time.perf_counter() - started
                log(
                    "export_progress",
                    shards_complete=completed,
                    shards_total=len(tasks),
                    images_complete=completed * plan.shard_images,
                    elapsed_seconds=elapsed,
                    images_per_second=completed * plan.shard_images / elapsed,
                    file_bytes=logical,
                    allocated_bytes=allocated,
                    latest_shard=shard,
                )
    log("verification_start")
    manifest["verified_samples"] = verify_samples(partial, plan, config.run.seed)
    zarr.open_group(str(partial), mode="r+").attrs.update(status="complete")
    manifest["checksummed_files"] = checksum_store(partial)
    logical, allocated = directory_bytes(partial)
    if max(logical, allocated) + 1_000_000 > plan.max_bytes:
        raise RuntimeError("Insufficient capacity left for final metadata")
    manifest.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        completed_at_utc=datetime.now(UTC).isoformat(),
        stored_file_bytes=logical,
        allocated_bytes=allocated,
    )
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (partial / "_SUCCESS").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(
                    (partial / "manifest.json").read_bytes()
                ).hexdigest(),
                "checksums_sha256": hashlib.sha256(
                    (partial / "checksums.sha256").read_bytes()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    partial.rename(destination)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(manifest, indent=2) + "\n")
    log("export_complete", **manifest, report=str(args.report))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/scratch"))
    parser.add_argument(
        "--allow-same-filesystem",
        action="store_true",
        help="Permit the selected dataset and results to share a filesystem",
    )
    parser.add_argument("--target-bytes", type=int, default=140_000_000_000)
    parser.add_argument("--max-bytes", type=int, default=150_000_000_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-images", type=int, default=8)
    parser.add_argument("--shard-images", type=int, default=256)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
