"""Bounded DDP storage benchmark with shard-aligned batches and streaming I/O."""

import argparse
import csv
import faulthandler
import gc
import hashlib
import json
import math
import os
import socket
import statistics
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import psutil  # type: ignore[import-untyped]
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

from vit_benchmark.config import load_config
from vit_benchmark.imaging import VisionTransformer
from vit_benchmark.run import DistributedContext
from vit_benchmark.telemetry import NvidiaSmiSampler
from vit_benchmark.zarr_stream import (
    SCRATCH,
    BatchLoader,
    ShardDataset,
    emit,
)


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * q))]


class CommState:
    def __init__(self) -> None:
        self.calls = 0
        self.bytes = 0
        self.completion_seconds = 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "gradient_collectives": self.calls,
            "gradient_bucket_bytes": self.bytes,
            "gradient_future_seconds_sum": self.completion_seconds,
        }


def communication_hook(
    state: CommState,
    bucket: dist.GradBucket,  # type: ignore[name-defined]
) -> torch.futures.Future[torch.Tensor]:
    started = time.perf_counter()
    state.calls += 1
    state.bytes += bucket.buffer().numel() * bucket.buffer().element_size()
    future = default_hooks.allreduce_hook(cast(dist.ProcessGroup, dist.group.WORLD), bucket)

    def finish(result: torch.futures.Future[torch.Tensor]) -> torch.Tensor:
        state.completion_seconds += time.perf_counter() - started
        return result.value()

    return cast(torch.futures.Future[torch.Tensor], future.then(finish))


class DevicePipeline:
    def __init__(self, loader: BatchLoader, device: torch.device) -> None:
        self.loader, self.device = loader, device
        self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self.pending: Any = None

    def enqueue(self) -> None:
        images, labels, stats = self.loader.next()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        with torch.cuda.stream(self.stream):
            start.record()
            x = images.to(self.device, non_blocking=True)
            y = labels.to(self.device, non_blocking=True)
            end.record()
        self.pending = (x, y, stats, start, end, images, labels)

    def next(self) -> Any:
        if self.pending is None:
            self.enqueue()
        item, self.pending = self.pending, None
        torch.cuda.current_stream().wait_event(item[4])
        item[0].record_stream(torch.cuda.current_stream())
        item[1].record_stream(torch.cuda.current_stream())
        return item

    def close(self) -> None:
        self.stream.synchronize()
        self.pending = None
        self.loader.close()


def train_update(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    pipeline: DevicePipeline,
    accumulation: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses, batches = [], []
    for micro in range(accumulation):
        item = pipeline.next()
        x, y = item[:2]
        sync = model.no_sync() if micro < accumulation - 1 else nullcontext()
        with sync, torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(x), y)
            torch.autograd.backward(loss / accumulation)
        losses.append(loss.detach())
        batches.append(item)
        if micro == accumulation - 1:
            optimizer.step()
        # CPU reads and next H2D are issued while current compute is queued.
        pipeline.enqueue()
    torch.cuda.synchronize(device)
    stats: dict[str, float] = {}
    for item in batches:
        for key, value in item[2].items():
            stats[key] = stats.get(key, 0) + value
        stats["h2d_seconds"] = stats.get("h2d_seconds", 0) + item[3].elapsed_time(item[4]) / 1000
    stats["step_seconds"] = time.perf_counter() - started
    loss_value = torch.stack(losses).mean().item()
    if not math.isfinite(loss_value):
        raise RuntimeError("Non-finite training loss")
    return loss_value, stats


def phase(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    pipeline: DevicePipeline,
    comm: CommState,
    context: DistributedContext,
    seconds: float,
    accumulation: int,
    name: str,
    output: Path,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    dist.barrier()
    before = comm.snapshot()
    started = time.perf_counter()
    samples = 0
    losses: list[float] = []
    sums: dict[str, float] = {}
    steps: list[float] = []
    with (output / f"steps-rank{context.rank}.jsonl").open("a") as log:
        while True:
            loss, stats = train_update(model, optimizer, pipeline, accumulation, context.device)
            steps.append(stats["step_seconds"])
            losses.append(loss)
            samples += pipeline.loader.batch_size * accumulation
            for key, value in stats.items():
                sums[key] = sums.get(key, 0) + value
            row = {
                "phase": name,
                "accumulation": accumulation,
                "update": len(steps),
                "loss": loss,
                **stats,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            # All ranks stop on the same optimizer boundary, even with a slow reader.
            elapsed = torch.tensor(time.perf_counter() - started, device=context.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if context.rank == 0 and (len(steps) == 1 or len(steps) % 10 == 0):
                emit(
                    "training_progress",
                    phase=name,
                    accumulation=accumulation,
                    updates=len(steps),
                    seconds=elapsed.item(),
                    loss=loss,
                    images_per_second=samples * context.world_size / elapsed.item(),
                )
            if elapsed.item() >= seconds:
                break
    dist.barrier()
    wall = time.perf_counter() - started
    after = comm.snapshot()
    return {
        "phase": name,
        "rank": context.rank,
        "hostname": socket.gethostname(),
        "accumulation": accumulation,
        "effective_global_batch": pipeline.loader.batch_size * context.world_size * accumulation,
        "samples": samples,
        "updates": len(steps),
        "seconds": wall,
        "loss": statistics.mean(losses),
        "step_p50_seconds": percentile(steps, 0.5),
        "step_p95_seconds": percentile(steps, 0.95),
        **sums,
        **{key: after[key] - before[key] for key in before},
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
        "rss_bytes": psutil.Process().memory_info().rss,
    }


def communication_probe(context: DistributedContext, output: Path, smoke: bool) -> None:
    rows = []
    for size in (25 * 1024**2, 344 * 1024**2):
        buffer = torch.ones(size // 4, device=context.device)
        for _ in range(2 if smoke else 5):
            dist.all_reduce(buffer)
            buffer.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        iterations = 5 if smoke else 20
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        for _ in range(iterations):
            dist.all_reduce(buffer)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000 / iterations
        row = {
            "bytes": size,
            "seconds_per_allreduce": seconds,
            "algorithmic_gb_s": size / seconds / 1e9,
            "bus_gb_s": size / seconds / 1e9 * 2 * (context.world_size - 1) / context.world_size,
        }
        rows.append(row)
        if context.is_primary:
            emit("allreduce_probe", **row)
    (output / f"allreduce-rank{context.rank}.json").write_text(json.dumps(rows, indent=2))
    del buffer
    torch.cuda.empty_cache()


def tune(
    dataset: ShardDataset,
    context: DistributedContext,
    output: Path,
    seed: int,
    smoke: bool,
    batch_size: int = 1024,
) -> int:
    trials = []
    for workers in (1, 2, 4):
        loader = BatchLoader(dataset, context.rank, context.world_size, workers, seed, batch_size)
        dist.barrier()
        started = time.perf_counter()
        batches = 2 if smoke else 4
        for _ in range(batches):
            batch = loader.next()
            del batch
        loader.close()  # Include the bounded extra prefetched batch in wall time.
        elapsed = torch.tensor(time.perf_counter() - started, device=context.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        trials.append({"workers": workers, "seconds": elapsed.item(), "consumed_batches": batches})
        if context.is_primary:
            emit("loader_tuning_trial", **trials[-1])
    selected = min(trials, key=lambda row: row["seconds"])["workers"]
    (output / f"tuning-rank{context.rank}.json").write_text(json.dumps(trials, indent=2))
    if context.is_primary:
        emit("loader_tuned", selected_workers=selected, trials=trials, prefetch_batches=1)
    return int(selected)


def validate(
    model: DDP, dataset: ShardDataset, context: DistributedContext, workers: int, seed: int
) -> dict[str, float]:
    loader = BatchLoader(dataset, context.rank, context.world_size, workers, seed)
    pipeline = DevicePipeline(loader, context.device)
    total = torch.zeros(2, device=context.device)
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(16384 // (1024 * context.world_size)):
                item = pipeline.next()
                total[0] += nn.functional.cross_entropy(model(item[0]), item[1], reduction="sum")
                total[1] += 1024
        dist.all_reduce(total)
        loss = (total[0] / total[1]).item()
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite validation")
        return {"loss": loss, "samples": int(total[1].item())}
    finally:
        pipeline.close()
        model.train()  # type: ignore[no-untyped-call]


def host_counters(output: Path, label: str, context: DistributedContext) -> None:
    if context.local_rank != 0:
        return
    infiniband: dict[str, str] = {}
    for path in Path("/sys/class/infiniband").glob("*/ports/*/hw_counters/*"):
        try:
            infiniband[str(path)] = path.read_text().strip()
        except OSError:
            continue
    counters = {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "infiniband": infiniband,
        "network": {
            name: value._asdict() for name, value in psutil.net_io_counters(pernic=True).items()
        },
        "disk": {
            name: value._asdict() for name, value in psutil.disk_io_counters(perdisk=True).items()
        },
        "memory": psutil.virtual_memory()._asdict(),
    }
    (output / f"host-{socket.gethostname()}-{label}.json").write_text(
        json.dumps(counters, indent=2)
    )


def verify_local_dataset(root: Path, seed: int) -> dict[str, Any]:
    """Validate a completed locally generated dataset and its metadata hashes."""
    marker = json.loads((root / "_SUCCESS").read_text())
    manifest_bytes = (root / "manifest.json").read_bytes()
    checksums = (root / "checksums.sha256").read_bytes()
    if (
        marker.get("status") != "complete"
        or marker.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
        or marker.get("checksums_sha256") != hashlib.sha256(checksums).hexdigest()
    ):
        raise ValueError("Local dataset receipt or metadata checksums differ")
    manifest = json.loads(manifest_bytes)
    for key, expected in {
        "status": "complete",
        "generator": "synthetic_imaging_v1",
        "seed": seed,
        "class_count": 12,
        "layout": "NCHW",
        "dtype": "float32",
        "train_compounds": list(range(80)),
        "validation_compounds": list(range(80, 100)),
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Local dataset differs: {key}")
    for key, expected in {
        "image_size": 224,
        "train_count": 185856,
        "validation_count": 46592,
        "chunk_images": 8,
        "shard_images": 256,
    }.items():
        if manifest["plan"].get(key) != expected:
            raise ValueError(f"Local dataset layout differs: {key}")
    return {
        "kind": "local_procedural_export",
        "manifest_sha256": marker["manifest_sha256"],
        "checksums_sha256": marker["checksums_sha256"],
        "regenerated_images_verified": manifest.get("regenerated_images_verified"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["scratch"], default="scratch")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=SCRATCH)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, choices=[1, 2, 4])
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 256 or args.batch_size % 256:
        parser.error("--batch-size must be a positive multiple of 256")
    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Eagerly initialize NCCL on the explicitly assigned device before any I/O.
    # A failed communicator cannot silently occupy the job until Slurm's limit.
    faulthandler.dump_traceback_later(90, exit=True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    faulthandler.cancel_dump_traceback_later()
    context = DistributedContext(
        rank=dist.get_rank(), world_size=dist.get_world_size(), local_rank=local_rank, device=device
    )
    emit("distributed_ready", rank=context.rank, device=str(device))
    if context.device.type != "cuda":
        raise ValueError("The disk-fed training benchmark requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    smoke = args.profile == "smoke"
    faulthandler.dump_traceback_later(60, exit=True)
    communication_probe(context, args.output, smoke)
    faulthandler.cancel_dump_traceback_later()
    config = load_config(args.config)
    assert config.imaging is not None
    if config.imaging.expected_world_size != context.world_size:
        raise ValueError("Config and actual world size differ")
    if config.imaging.micro_batch_size_per_rank != args.batch_size:
        raise ValueError("Config and requested batch size differ")
    identity = verify_local_dataset(args.dataset, config.run.seed)
    dist.barrier()
    host_counters(args.output, "start", context)
    dataset = ShardDataset(args.dataset)
    validation = ShardDataset(args.dataset, "validation")
    if dataset.count != 185856 or validation.count != 46592:
        raise ValueError("Unexpected dataset counts")
    smoke = args.profile == "smoke"
    sampler = NvidiaSmiSampler(
        artifact_path=args.output / f"telemetry-rank{context.rank}.jsonl",
        rank=context.rank,
        stdout=True,
    )
    sampler.start()
    rank_results = []
    validations = []
    started = time.perf_counter()
    try:
        workers = args.workers or tune(
            dataset, context, args.output, config.run.seed, smoke, args.batch_size
        )
        for accumulation in (1, 2):
            torch.manual_seed(config.run.seed)
            model = DDP(
                VisionTransformer(config.imaging).to(context.device),
                device_ids=[context.local_rank],
            )
            comm = CommState()
            model.register_comm_hook(comm, communication_hook)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.imaging.learning_rate,
                weight_decay=config.imaging.weight_decay,
            )
            loader = BatchLoader(
                dataset, context.rank, context.world_size, workers, config.run.seed, args.batch_size
            )
            pipeline = DevicePipeline(loader, context.device)
            try:
                torch.cuda.reset_peak_memory_stats()
                for name, seconds in [
                    ("warmup", 5 if smoke else 120),
                    *[(f"trial-{i}", 5 if smoke else 180) for i in range(3)],
                ]:
                    sampler.set_phase(f"accum{accumulation}-{name}")
                    measured = phase(
                        model,
                        optimizer,
                        pipeline,
                        comm,
                        context,
                        seconds,
                        accumulation,
                        name,
                        args.output,
                    )
                    rank_results.append(measured)
                    emit("rank_phase", **measured)
                pipeline.close()
                checked = validate(model, validation, context, workers, config.run.seed)
                validations.append({"accumulation": accumulation, **checked})
                emit("validation_complete", rank=context.rank, accumulation=accumulation, **checked)
                if args.trace and accumulation == 1:
                    trace_dir = args.output / "traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    profiler_loader = BatchLoader(
                        dataset,
                        context.rank,
                        context.world_size,
                        workers,
                        config.run.seed,
                        args.batch_size,
                    )
                    profiler_pipeline = DevicePipeline(profiler_loader, context.device)
                    try:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler:
                            train_update(model, optimizer, profiler_pipeline, 1, context.device)
                        profiler.export_chrome_trace(str(trace_dir / f"rank{context.rank}.json"))
                    finally:
                        profiler_pipeline.close()
            finally:
                pipeline.close()
            del optimizer, model, pipeline, loader, comm
            gc.collect()
            torch.cuda.empty_cache()
        host_counters(args.output, "end", context)
        telemetry = sampler.stop()
        if not telemetry.sample_count or telemetry.diagnostic:
            raise RuntimeError(f"Missing telemetry: {telemetry.diagnostic}")
        result = {
            "rank": context.rank,
            "hostname": socket.gethostname(),
            "source": args.source,
            "world_size": context.world_size,
            "batch_per_gpu": args.batch_size,
            "dataset_path": str(args.dataset),
            "dataset_identity": identity,
            "validation_batch_per_gpu": 1024,
            "runtime": sampler.runtime.model_dump(mode="json") if sampler.runtime else None,
            "provider": config.provider.model_dump(),
            "gpu_name": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE", "default"),
            "workers": workers,
            "cpu_prefetch_batches": 1,
            "image_cache": "none",
            "scratch_io": "O_DIRECT" if args.source == "scratch" else "not_applicable",
            "elapsed_seconds": time.perf_counter() - started,
            "phases": rank_results,
            "validation": validations,
            "telemetry": telemetry.model_dump(mode="json"),
        }
        (args.output / f"rank{context.rank}.json").write_text(json.dumps(result, indent=2))
        ranks: list[Any] = [None] * context.world_size
        dist.all_gather_object(ranks, result)
        if context.is_primary:
            for rank_result in ranks:
                (args.output / f"rank{rank_result['rank']}.json").write_text(
                    json.dumps(rank_result, indent=2)
                )
            rows = []
            for index, phase_result in enumerate(rank_results):
                phases = [rank["phases"][index] for rank in ranks]
                seconds = max(p["seconds"] for p in phases)
                row = {
                    "source": args.source,
                    "gpus": context.world_size,
                    "accumulation": phase_result["accumulation"],
                    "phase": phase_result["phase"],
                    "effective_global_batch": phase_result["effective_global_batch"],
                    "seconds": seconds,
                    "images_per_second": sum(p["samples"] for p in phases) / seconds,
                    "image_read_gb_s": sum(p["image_bytes"] for p in phases) / seconds / 1e9,
                    "physical_read_bytes": sum(p["physical_bytes"] for p in phases),
                    "image_bytes": sum(p["image_bytes"] for p in phases),
                    "gradient_collectives_per_rank": phase_result["gradient_collectives"],
                    "mean_loss": statistics.mean(p["loss"] for p in phases),
                }
                if args.source == "scratch" and row["physical_read_bytes"] < row["image_bytes"]:
                    raise RuntimeError("Physical I/O acceptance check failed")
                rows.append(row)
                emit("phase_summary", **row)
            with (args.output / "summary.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = {
                "status": "complete",
                "source": args.source,
                "profile": args.profile,
                "gpus": context.world_size,
                "ranks": len(ranks),
                "batch_per_gpu": args.batch_size,
                "elapsed_seconds": max(rank["elapsed_seconds"] for rank in ranks),
            }
            (args.output / "_SUCCESS").write_text(json.dumps(summary, indent=2))
            emit("job_verified", **summary)
        dist.barrier()
    finally:
        sampler.stop()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
