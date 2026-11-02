"""ViT measurement and runtime records."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

RUN_RECORD_SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"


class ProviderConfiguration(BaseModel):
    """Exact rented configuration, recorded for every measured run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    instance_type: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: PositiveInt
    node_count: PositiveInt
    network_description: str = Field(min_length=1)
    storage_description: str = Field(min_length=1)


class TrialMeasurement(BaseModel):
    """Metrics for one timed measurement window."""

    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float = Field(gt=0)
    optimizer_updates: PositiveInt
    samples_processed: PositiveInt
    throughput_samples_per_second: float = Field(gt=0)
    step_time_median_ms: float = Field(gt=0)
    step_time_p95_ms: float = Field(gt=0)
    data_wait_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int = Field(ge=0)
    validation_loss: float = Field(ge=0)


class GpuTelemetry(BaseModel):
    """Best-effort per-rank training telemetry summary."""

    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=0)
    utilization_percent_mean: float | None = Field(default=None, ge=0, le=100)
    memory_used_mib_max: float | None = Field(default=None, ge=0)
    utilization_percent_max: float | None = Field(default=None, ge=0, le=100)
    utilization_percent_p95: float | None = Field(default=None, ge=0, le=100)
    cpu_utilization_percent_max: float | None = Field(default=None, ge=0, le=100)
    host_memory_used_mib_max: float | None = Field(default=None, ge=0)
    rank: int | None = Field(default=None, ge=0)
    hostname: str | None = None
    raw_artifact_path: str | None = None
    diagnostic: str | None = None
    query_retry_count: int = Field(default=0, ge=0)


class RuntimeEnvironment(BaseModel):
    """Versioned hardware/runtime identity captured from one training rank."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=0)
    hostname: str = Field(min_length=1)
    nvidia_driver_version: str | None = None
    nvidia_smi_version: str | None = None
    cuda_driver_version: str | None = None
    pytorch_version: str = Field(min_length=1)
    pytorch_cuda_version: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class BenchmarkRunRecord(BaseModel):
    """The interoperable record emitted for each completed benchmark run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = RUN_RECORD_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    mode: Literal["gpu_only", "real_pipeline"]
    profile: str = Field(min_length=1)
    started_at_utc: datetime
    provider: ProviderConfiguration
    world_size: PositiveInt
    seed: int
    requested_precision: Literal["fp32", "bf16"]
    effective_precision: Literal["fp32", "bf16"]
    global_batch_size: PositiveInt
    micro_batch_size_per_rank: PositiveInt
    gradient_accumulation_steps: PositiveInt
    generator: str = Field(min_length=1)
    generator_parameters: dict[str, int | float | str | bool]
    trials: list[TrialMeasurement] = Field(min_length=1)
    fixed_job: TrialMeasurement
    gpu_telemetry: GpuTelemetry
    telemetry_by_rank: list[GpuTelemetry] = Field(default_factory=list)
    runtime_environments: list[RuntimeEnvironment] = Field(default_factory=list)
    checkpoint_seconds: float = Field(ge=0)
    export_seconds: float = Field(ge=0)
    diagnostics: list[str]
