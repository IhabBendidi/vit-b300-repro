"""Synthetic compound-classification ViT workload."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from vit_benchmark.config import BenchmarkConfig, ImagingSettings
from vit_benchmark.contracts import BenchmarkRunRecord, TrialMeasurement
from vit_benchmark.results import write_record
from vit_benchmark.run import (
    DistributedContext,
    Precision,
    _ensure_artifact_budget,
    _log,
    _percentile,
    _synchronize_max,
    collect_rank_telemetry,
    initialize_distributed,
    resolve_precision,
)
from vit_benchmark.telemetry import NvidiaSmiSampler


@dataclass(frozen=True)
class SyntheticImageSpec:
    seed: int
    dataset_size: int
    image_size: int
    class_count: int
    compound_ids: tuple[int, ...]


class SyntheticImagingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Indexed, procedural microscopy-like images with compound-linked labels."""

    def __init__(self, spec: SyntheticImageSpec) -> None:
        self.spec = spec
        axis = torch.linspace(-1.0, 1.0, spec.image_size)
        self.grid_y, self.grid_x = torch.meshgrid(axis, axis, indexing="ij")

    def __len__(self) -> int:
        return self.spec.dataset_size

    def __getitem__(self, sample_id: int) -> tuple[Tensor, Tensor]:
        compound_id = self.spec.compound_ids[sample_id % len(self.spec.compound_ids)]
        label = compound_id % self.spec.class_count
        generator = torch.Generator().manual_seed(self.spec.seed * 1_000_003 + sample_id)
        phase = (compound_id + 1) * 0.173
        frequency = 2.0 + (label % 4)
        stripe_x = torch.sin(frequency * torch.pi * self.grid_x + phase)
        stripe_y = torch.cos((frequency + 0.5) * torch.pi * self.grid_y - phase)
        center_x = ((label % 3) - 1) * 0.35 + (compound_id % 5 - 2) * 0.025
        center_y = ((label // 3) - 1) * 0.35 + (compound_id % 7 - 3) * 0.02
        blob = torch.exp(-12 * ((self.grid_x - center_x) ** 2 + (self.grid_y - center_y) ** 2))
        rings = torch.sin((label + 2) * torch.pi * torch.sqrt(self.grid_x**2 + self.grid_y**2))
        image = torch.stack((stripe_x, stripe_y, blob + 0.25 * rings))
        image += 0.08 * torch.randn(image.shape, generator=generator)
        image += torch.rand((), generator=generator) * 0.08
        return image.float(), torch.tensor(label, dtype=torch.long)


def imaging_parameters(
    spec: SyntheticImageSpec, settings: ImagingSettings
) -> dict[str, int | float | str | bool]:
    return {
        "dataset_size": settings.dataset_size,
        "validation_sample_count": settings.validation_sample_count,
        "image_size": spec.image_size,
        "class_count": spec.class_count,
        "compound_count": settings.compound_count,
        "train_compound_count": settings.train_compound_count,
        "storage": "indexed_on_the_fly",
        "pad_training_batches": settings.pad_training_batches,
    }


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, attention_heads: int, mlp_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size, eps=1e-6)
        self.attention = nn.MultiheadAttention(embedding_size, attention_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_size), nn.GELU(), nn.Linear(mlp_size, embedding_size)
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + attended
        return values + cast(Tensor, self.mlp(self.norm2(values)))


class VisionTransformer(nn.Module):
    def __init__(self, settings: ImagingSettings) -> None:
        super().__init__()
        patch_count = (settings.image_size // settings.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, settings.embedding_size, settings.patch_size, settings.patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, settings.embedding_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, settings.embedding_size)
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    settings.embedding_size, settings.attention_heads, settings.mlp_size
                )
                for _ in range(settings.transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(settings.embedding_size, eps=1e-6)
        self.head = nn.Linear(settings.embedding_size, settings.class_count)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.head(self.embed(images)))

    def embed(self, images: Tensor) -> Tensor:
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1) + self.position_embedding
        tokens = self.blocks(tokens)
        return cast(Tensor, self.norm(tokens[:, 0]))


def _settings(config: BenchmarkConfig) -> ImagingSettings:
    if config.imaging is None:
        raise RuntimeError("imaging runner requires [imaging] settings")
    return config.imaging


def _make_loader(
    config: BenchmarkConfig, context: DistributedContext, validation: bool = False
) -> DataLoader[tuple[Tensor, Tensor]]:
    settings = _settings(config)
    compound_ids = (
        tuple(range(settings.train_compound_count, settings.compound_count))
        if validation
        else tuple(range(settings.train_compound_count))
    )
    spec = SyntheticImageSpec(
        seed=config.run.seed,
        dataset_size=settings.validation_sample_count
        if validation
        else settings.dataset_size - settings.validation_sample_count,
        image_size=settings.image_size,
        class_count=settings.class_count,
        compound_ids=compound_ids,
    )
    dataset = SyntheticImagingDataset(spec)
    sampler: DistributedSampler[tuple[Tensor, Tensor]] | None = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.run.seed,
        )
        if context.world_size > 1
        else None
    )
    if sampler is not None and not validation and settings.pad_training_batches:
        # Keep the fixed dataset while repeating samples when a rank's shard is
        # smaller than one full training batch (1000 samples on sixteen GPUs).
        sampler.num_samples = max(sampler.num_samples, settings.micro_batch_size_per_rank)
        sampler.total_size = sampler.num_samples * context.world_size
    return DataLoader(
        dataset,
        batch_size=settings.micro_batch_size_per_rank,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=settings.num_workers,
        drop_last=not validation,
        pin_memory=context.device.type == "cuda",
    )


def _make_model(config: BenchmarkConfig, context: DistributedContext) -> nn.Module:
    torch.manual_seed(config.run.seed)
    model: nn.Module = VisionTransformer(_settings(config)).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[context.local_rank] if context.device.type == "cuda" else None
        )
    return model


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_batch: Callable[[], tuple[Tensor, Tensor]],
    device: torch.device,
    accumulation: int,
    precision: Precision,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    data_wait, loss_value = 0.0, 0.0
    for micro_step in range(accumulation):
        started = time.perf_counter()
        images, labels = next_batch()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        data_wait += time.perf_counter() - started
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        )
        gradient_sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and micro_step != accumulation - 1
            else nullcontext()
        )
        with gradient_sync, autocast:
            logits = cast(Tensor, model(images))
            loss = nn.functional.cross_entropy(logits, labels)
            torch.autograd.backward(loss / accumulation)
        loss_value += float(loss.detach().cpu())
    optimizer.step()
    return data_wait, loss_value / accumulation


def _validation_loss(
    model: nn.Module, config: BenchmarkConfig, context: DistributedContext, precision: Precision
) -> float:
    total_loss, total_samples = (
        torch.zeros((), device=context.device),
        torch.zeros((), device=context.device),
    )
    model.eval()
    with torch.no_grad():
        for images, labels in _make_loader(config, context, validation=True):
            images, labels = images.to(context.device), labels.to(context.device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with autocast:
                logits = cast(Tensor, model(images))
                total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").float()
            total_samples += labels.numel()
    if context.world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_samples).cpu())


def _run_trial(
    config: BenchmarkConfig,
    context: DistributedContext,
    seconds: float | None,
    updates: int | None,
    gpu_only: bool,
    accumulation: int,
    precision: Precision,
    retain_state: bool = False,
    phase: str = "trial",
) -> tuple[TrialMeasurement, dict[str, Tensor] | None]:
    settings = _settings(config)
    _log(context, "phase_start", phase=phase, seconds=seconds, target_updates=updates)
    model = _make_model(config, context)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loader = _make_loader(config, context)
    iterator = iter(loader)
    preloaded: tuple[Tensor, Tensor] | None = None
    if gpu_only:
        source = cast(tuple[Tensor, Tensor], next(iterator))
        preloaded = cast(tuple[Tensor, Tensor], tuple(value.to(context.device) for value in source))

    def next_batch() -> tuple[Tensor, Tensor]:
        nonlocal iterator
        if preloaded is not None:
            return preloaded
        try:
            return cast(tuple[Tensor, Tensor], next(iterator))
        except StopIteration:
            iterator = iter(loader)
            return cast(tuple[Tensor, Tensor], next(iterator))

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.cuda.synchronize(context.device)
    started, steps, step_times, waits = time.perf_counter(), 0, [], []
    elapsed, last_log = 0.0, 0.0
    while (seconds is not None and elapsed < seconds) or (updates is not None and steps < updates):
        step_started = time.perf_counter()
        wait, loss = _step(model, optimizer, next_batch, context.device, accumulation, precision)
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        step_times.append(_synchronize_max(time.perf_counter() - step_started, context))
        waits.append(_synchronize_max(wait, context))
        steps += 1
        elapsed = _synchronize_max(time.perf_counter() - started, context)
        if steps == 1 or elapsed - last_log >= 5:
            _log(
                context,
                "progress",
                phase=phase,
                optimizer_updates=steps,
                elapsed_seconds=elapsed,
                throughput_samples_per_second=steps * settings.global_batch_size / elapsed,
                step_time_ms=step_times[-1] * 1000,
                training_loss=loss,
                data_wait_seconds=sum(waits),
                peak_gpu_memory_bytes=(
                    torch.cuda.max_memory_allocated(context.device)
                    if context.device.type == "cuda"
                    else 0
                ),
            )
            last_log = elapsed
    peak_memory = (
        torch.cuda.max_memory_allocated(context.device) if context.device.type == "cuda" else 0
    )
    samples = steps * settings.global_batch_size
    _log(context, "validation_start", phase=phase, optimizer_updates=steps)
    measurement = TrialMeasurement(
        elapsed_seconds=elapsed,
        optimizer_updates=steps,
        samples_processed=samples,
        throughput_samples_per_second=samples / elapsed,
        step_time_median_ms=_percentile(step_times, 0.5) * 1000,
        step_time_p95_ms=_percentile(step_times, 0.95) * 1000,
        data_wait_seconds=sum(waits),
        peak_gpu_memory_bytes=int(_synchronize_max(float(peak_memory), context)),
        validation_loss=_validation_loss(model, config, context, precision),
    )
    _log(context, "phase_complete", phase=phase, **measurement.model_dump(mode="json"))
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    state = (
        {name: value.detach().cpu() for name, value in raw_model.state_dict().items()}
        if retain_state and context.is_primary
        else None
    )
    return measurement, state


def execute_imaging(config: BenchmarkConfig, profile_name: str) -> Path | None:
    context, settings = initialize_distributed(), _settings(config)
    if context.world_size != settings.expected_world_size:
        raise RuntimeError(
            f"expected world size {settings.expected_world_size}, got {context.world_size}"
        )
    accumulation, precision = (
        settings.gradient_accumulation_steps(context.world_size),
        resolve_precision(config, context),
    )
    started_at = datetime.now(UTC)
    _log(
        context,
        "run_start",
        world_size=context.world_size,
        precision=precision,
        gradient_accumulation_steps=accumulation,
        config=config.model_dump(mode="json"),
    )
    sampler = NvidiaSmiSampler(
        artifact_path=config.run.artifact_dir
        / f"{config.run.run_id}.telemetry.rank{context.rank}.jsonl",
        rank=context.rank,
        stdout=True,
    )
    sampler.start()
    try:
        sampler.set_phase("warmup")
        _run_trial(
            config,
            context,
            config.timing.warmup_seconds,
            None,
            config.run.mode == "gpu_only",
            accumulation,
            precision,
            phase="warmup",
        )
        trials = []
        for index in range(config.timing.measurement_repetitions):
            sampler.set_phase(f"measurement_{index + 1}")
            trials.append(
                _run_trial(
                    config,
                    context,
                    config.timing.measurement_seconds,
                    None,
                    config.run.mode == "gpu_only",
                    accumulation,
                    precision,
                    phase=f"measurement_{index + 1}",
                )[0]
            )
        sampler.set_phase("fixed_job")
        fixed_job, state = _run_trial(
            config,
            context,
            None,
            config.timing.fixed_job_updates,
            config.run.mode == "gpu_only",
            accumulation,
            precision,
            retain_state=True,
            phase="fixed_job",
        )
        telemetry, telemetry_by_rank, runtime_environments = collect_rank_telemetry(
            sampler, context
        )
        if not context.is_primary:
            return None
        if state is None:
            raise RuntimeError("rank zero did not retain the fixed-job model state")
        config.run.artifact_dir.mkdir(parents=True, exist_ok=True)
        _ensure_artifact_budget(config.run.artifact_dir, state, config.run.artifact_max_bytes)
        checkpoint_started = time.perf_counter()
        torch.save(state, config.run.artifact_dir / f"{config.run.run_id}.pt")
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        export_started = time.perf_counter()
        (config.run.artifact_dir / f"{config.run.run_id}.export.json").write_text(
            '{"format":"vit-benchmark-synthetic-imaging-v1"}\n', encoding="utf-8"
        )
        export_seconds = time.perf_counter() - export_started
        spec = SyntheticImageSpec(
            config.run.seed,
            settings.dataset_size,
            settings.image_size,
            settings.class_count,
            tuple(range(settings.train_compound_count)),
        )
        diagnostics = [telemetry.diagnostic] if telemetry.diagnostic else []
        record = BenchmarkRunRecord(
            run_id=config.run.run_id,
            workload=config.run.workload,
            mode=config.run.mode,
            profile=profile_name,
            started_at_utc=started_at,
            provider=config.provider,
            world_size=context.world_size,
            seed=config.run.seed,
            requested_precision=config.run.precision,
            effective_precision=precision,
            global_batch_size=settings.global_batch_size,
            micro_batch_size_per_rank=settings.micro_batch_size_per_rank,
            gradient_accumulation_steps=accumulation,
            generator="synthetic_imaging_v1",
            generator_parameters=imaging_parameters(spec, settings),
            trials=trials,
            fixed_job=fixed_job,
            gpu_telemetry=telemetry,
            telemetry_by_rank=telemetry_by_rank,
            runtime_environments=runtime_environments,
            checkpoint_seconds=checkpoint_seconds,
            export_seconds=export_seconds,
            diagnostics=diagnostics,
        )
        path = write_record(record, config.run.output_dir)
        throughputs = sorted(trial.throughput_samples_per_second for trial in trials)
        median = throughputs[len(throughputs) // 2]
        _log(
            context,
            "run_complete",
            result_path=str(path),
            median_throughput_samples_per_second=median,
            throughput_range=[throughputs[0], throughputs[-1]],
            record=record.model_dump(mode="json"),
        )
        return path
    finally:
        if sampler._thread is not None and sampler._thread.is_alive():
            sampler.stop()
        if context.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    import argparse

    from vit_benchmark.config import load_config

    parser = argparse.ArgumentParser(description="ViT training with one batch cached on each GPU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"])
    args = parser.parse_args()
    config = load_config(args.config, args.profile)
    if config.run.mode != "gpu_only":
        parser.error("This command requires mode = 'gpu_only'")
    execute_imaging(config, args.profile or "configured")


if __name__ == "__main__":
    main()
