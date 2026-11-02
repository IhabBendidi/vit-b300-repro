"""TOML configuration parsing and profile defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from vit_benchmark.contracts import ProviderConfiguration


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    workload: Literal["synthetic_imaging_vit"] = "synthetic_imaging_vit"
    mode: Literal["gpu_only", "real_pipeline"]
    seed: int = 20261005
    output_dir: Path = Path("results")
    artifact_dir: Path = Path("artifacts")
    artifact_max_bytes: PositiveInt = 512 * 1024 * 1024
    precision: Literal["fp32", "bf16"] = "fp32"
    dataset_path: Path | None = None


class ImagingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["local_smoke", "provider_full"] = "local_smoke"
    dataset_size: PositiveInt
    validation_sample_count: PositiveInt
    class_count: PositiveInt
    compound_count: PositiveInt
    train_compound_count: PositiveInt
    image_size: PositiveInt
    patch_size: PositiveInt
    embedding_size: PositiveInt
    transformer_depth: PositiveInt
    attention_heads: PositiveInt
    mlp_size: PositiveInt
    global_batch_size: PositiveInt
    micro_batch_size_per_rank: PositiveInt
    expected_world_size: PositiveInt = 1
    num_workers: int = Field(default=0, ge=0)
    pad_training_batches: bool = False
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)

    def gradient_accumulation_steps(self, world_size: int) -> int:
        denominator = world_size * self.micro_batch_size_per_rank
        if self.global_batch_size % denominator != 0:
            raise ValueError(
                "global_batch_size must be divisible by world_size * micro_batch_size_per_rank"
            )
        return self.global_batch_size // denominator


class TimingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warmup_seconds: float = Field(gt=0)
    measurement_seconds: float = Field(gt=0)
    measurement_repetitions: PositiveInt = 3
    fixed_job_updates: PositiveInt


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: RunSettings
    provider: ProviderConfiguration
    imaging: ImagingSettings | None = None
    timing: TimingSettings

    @model_validator(mode="after")
    def validate_profile_shape(self) -> BenchmarkConfig:
        settings = self.imaging
        if settings is None:
            raise ValueError("configuration requires [imaging]")
        if settings.validation_sample_count >= settings.dataset_size:
            raise ValueError("validation_sample_count must be less than dataset_size")
        if self.provider.gpu_count != settings.expected_world_size:
            raise ValueError("provider.gpu_count must equal imaging.expected_world_size")
        settings.gradient_accumulation_steps(settings.expected_world_size)
        imaging = self.imaging
        assert imaging is not None
        if imaging.train_compound_count >= imaging.compound_count:
            raise ValueError("train_compound_count must be less than compound_count")
        if imaging.image_size % imaging.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if imaging.embedding_size % imaging.attention_heads != 0:
            raise ValueError("embedding_size must be divisible by attention_heads")
        if imaging.profile == "provider_full":
            if (imaging.dataset_size, imaging.validation_sample_count) != (20_000, 4_000):
                raise ValueError(
                    "provider_full imaging requires 20,000 total samples and 4,000 validation samples"
                )
            architecture = (
                imaging.image_size,
                imaging.patch_size,
                imaging.embedding_size,
                imaging.transformer_depth,
                imaging.attention_heads,
                imaging.mlp_size,
            )
            if architecture != (224, 16, 768, 12, 12, 3072):
                raise ValueError("provider_full imaging requires the fixed ViT-B/16 architecture")
            if (imaging.class_count, imaging.compound_count, imaging.train_compound_count) != (
                12,
                100,
                80,
            ):
                raise ValueError(
                    "provider_full imaging requires 12 classes and an 80/20 compound split"
                )
            if self.run.precision != "bf16":
                raise ValueError("provider_full requires precision = 'bf16'")
        elif self.run.precision != "fp32":
            raise ValueError("local_smoke requires precision = 'fp32'")
        return self


PROFILE_OVERRIDES: dict[str, dict[str, int | float]] = {
    "smoke": {
        "warmup_seconds": 5.0,
        "measurement_seconds": 10.0,
        "measurement_repetitions": 3,
        "fixed_job_updates": 50,
    },
    "full": {
        "warmup_seconds": 600.0,
        "measurement_seconds": 1200.0,
        "measurement_repetitions": 3,
        "fixed_job_updates": 5000,
    },
}


def load_config(path: Path | str, profile: str | None = None) -> BenchmarkConfig:
    """Load a checked TOML file, optionally replacing only timing values."""

    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if profile is not None:
        if profile not in PROFILE_OVERRIDES:
            raise ValueError(f"Unknown timing profile {profile!r}; choose smoke or full")
        raw["timing"] = PROFILE_OVERRIDES[profile]
    return BenchmarkConfig.model_validate(raw)
