"""Distributed runtime helpers used by the ViT benchmark."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor

from vit_benchmark.config import BenchmarkConfig
from vit_benchmark.contracts import GpuTelemetry, RuntimeEnvironment
from vit_benchmark.telemetry import NvidiaSmiSampler

Precision = Literal["fp32", "bf16"]


def _log(context: DistributedContext, event: str, **values: object) -> None:
    if context.is_primary:
        sys.stdout.write(
            json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **values})
            + "\n"
        )
        sys.stdout.flush()


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def resolve_precision(config: BenchmarkConfig, context: DistributedContext) -> Precision:
    """Validate and return the effective training precision."""

    if config.run.precision == "fp32":
        return "fp32"
    if context.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision = 'bf16' requires a CUDA GPU with BF16 support")
    return "bf16"


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedContext(
        rank=rank, world_size=world_size, local_rank=local_rank, device=device
    )


def _synchronize_max(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    value_tensor = torch.tensor(value, device=context.device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    return value_tensor.item()


def collect_rank_telemetry(
    sampler: NvidiaSmiSampler, context: DistributedContext
) -> tuple[GpuTelemetry, list[GpuTelemetry], list[RuntimeEnvironment]]:
    """Stop local telemetry and gather compact summaries from every training rank."""

    local_summary = sampler.stop()
    local_runtime = sampler.runtime
    if local_runtime is None:
        raise RuntimeError("telemetry sampler did not capture runtime metadata")
    payload: dict[str, object] = {
        "telemetry": local_summary.model_dump(mode="json"),
        "runtime": local_runtime.model_dump(mode="json"),
    }
    gathered: list[object] = [payload]
    if context.world_size > 1:
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, payload)
    payloads: list[dict[str, object]] = []
    for item in gathered:
        if not isinstance(item, dict):
            raise RuntimeError("invalid gathered telemetry payload")
        payloads.append(cast(dict[str, object], item))
    summaries = sorted(
        (GpuTelemetry.model_validate(item["telemetry"]) for item in payloads),
        key=lambda item: item.rank if item.rank is not None else -1,
    )
    runtimes = sorted(
        (RuntimeEnvironment.model_validate(item["runtime"]) for item in payloads),
        key=lambda item: item.rank,
    )
    primary = next((item for item in summaries if item.rank == 0), summaries[0])
    return primary, summaries, runtimes


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _ensure_artifact_budget(
    artifact_dir: Path, state: dict[str, Tensor], maximum_bytes: int
) -> None:
    existing_bytes = sum(path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file())
    state_bytes = sum(value.numel() * value.element_size() for value in state.values())
    if existing_bytes + state_bytes > maximum_bytes:
        raise RuntimeError(
            f"checkpoint would exceed artifact_max_bytes ({maximum_bytes} bytes); "
            "use a larger provider-run limit or clear old artifacts"
        )
