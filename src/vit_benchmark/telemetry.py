"""Portable, per-rank training telemetry and NVIDIA runtime identity."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]
import torch

from vit_benchmark.contracts import GpuTelemetry, RuntimeEnvironment

GPU_QUERY = (
    "index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "power.draw,power.limit,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current"
)


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def collect_runtime_environment(rank: int) -> RuntimeEnvironment:
    """Capture driver/runtime identity without failing a CPU run."""

    hostname, diagnostics = platform.node() or os.uname().nodename, []
    if shutil.which("nvidia-smi") is None:
        diagnostics.append("nvidia-smi not found")
        return RuntimeEnvironment(
            rank=rank,
            hostname=hostname,
            pytorch_version=torch.__version__,
            pytorch_cuda_version=torch.version.cuda,
            diagnostics=diagnostics,
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        driver = rows[0][0].strip() if rows else None
        uuids = [row[1].strip() for row in rows if len(row) > 1]
        version_output = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
        version = next((line.strip() for line in version_output if "NVIDIA-SMI" in line), None)
        cuda_version = (
            match.group(1)
            if version and (match := re.search(r"CUDA Version:\s*([^\s|]+)", version))
            else None
        )
        return RuntimeEnvironment(
            rank=rank,
            hostname=hostname,
            nvidia_driver_version=driver,
            nvidia_smi_version=version,
            cuda_driver_version=cuda_version,
            pytorch_version=torch.__version__,
            pytorch_cuda_version=torch.version.cuda,
            gpu_uuids=uuids,
        )
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        diagnostics.append(f"nvidia-smi metadata failed: {error}")
        return RuntimeEnvironment(
            rank=rank,
            hostname=hostname,
            pytorch_version=torch.__version__,
            pytorch_cuda_version=torch.version.cuda,
            diagnostics=diagnostics,
        )


@dataclass
class NvidiaSmiSampler:
    """Write a one-second JSONL stream and return a compact rank summary."""

    artifact_path: Path | None = None
    rank: int = 0
    interval_seconds: float = 1.0
    stdout: bool = False
    phase: str = "setup"
    samples: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: str | None = None
    query_retry_count: int = 0
    runtime: RuntimeEnvironment | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        self.runtime = collect_runtime_environment(self.rank)
        psutil.cpu_percent(percpu=True)
        if self.artifact_path is not None:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def stop(self) -> GpuTelemetry:
        self._stop.set()
        if self._thread is not None:
            # A query may use its five-second timeout twice before failing.
            self._thread.join(timeout=self.interval_seconds + 12)
            if self._thread.is_alive():
                self.diagnostic = "telemetry worker did not stop within its bounded timeout"
        gpu_utilizations = [
            gpu["utilization_gpu_percent"]
            for sample in self.samples
            for gpu in sample["gpus"]
            if gpu.get("utilization_gpu_percent") is not None
        ]
        gpu_memories = [
            gpu["memory_used_mib"]
            for sample in self.samples
            for gpu in sample["gpus"]
            if gpu.get("memory_used_mib") is not None
        ]
        cpu = [sample["host"]["cpu_utilization_percent"] for sample in self.samples]
        ram = [sample["host"]["memory_used_mib"] for sample in self.samples]
        return GpuTelemetry(
            sample_count=len(self.samples),
            utilization_percent_mean=(
                sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else None
            ),
            utilization_percent_max=max(gpu_utilizations) if gpu_utilizations else None,
            utilization_percent_p95=_percentile(gpu_utilizations, 0.95)
            if gpu_utilizations
            else None,
            memory_used_mib_max=max(gpu_memories) if gpu_memories else None,
            cpu_utilization_percent_max=max(cpu) if cpu else None,
            host_memory_used_mib_max=max(ram) if ram else None,
            rank=self.rank,
            hostname=self.runtime.hostname if self.runtime else None,
            raw_artifact_path=str(self.artifact_path) if self.artifact_path else None,
            diagnostic=self.diagnostic,
            query_retry_count=self.query_retry_count,
        )

    def _gpu_sample(self) -> list[dict[str, Any]]:
        if shutil.which("nvidia-smi") is None:
            self.diagnostic = "nvidia-smi not found"
            return []
        try:
            command = ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
            # CUDA indices are relative to Slurm's visible-device mask. Query
            # only this worker's assigned GPU, including for single-GPU jobs.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if visible[0] and local_rank < len(visible):
                command.append(f"--id={visible[local_rank]}")
            for attempt in range(2):
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        check=True,
                        text=True,
                        timeout=5,
                    )
                    break
                except subprocess.TimeoutExpired:
                    if attempt:
                        raise
                    self.query_retry_count += 1
            names = GPU_QUERY.split(",")
            return [
                {
                    {
                        "utilization.gpu": "utilization_gpu_percent",
                        "memory.used": "memory_used_mib",
                    }.get(name, name): (
                        _number(value)
                        if name not in {"uuid", "name", "driver_version"}
                        else value.strip()
                    )
                    for name, value in zip(names, line.split(","), strict=True)
                }
                for line in result.stdout.splitlines()
                if line.strip()
            ]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.diagnostic = f"nvidia-smi sampling failed: {error}"
            return []

    def _collect(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                phase = self.phase
            memory, frequency = psutil.virtual_memory(), psutil.cpu_freq()
            disk_io = psutil.disk_io_counters()
            try:
                swap_used_mib = psutil.swap_memory().used / (1024 * 1024)
            except OSError:
                # Some restricted local environments do not expose swap counters.
                swap_used_mib = None
            sample = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "rank": self.rank,
                "hostname": self.runtime.hostname if self.runtime else platform.node(),
                "phase": phase,
                "gpus": self._gpu_sample(),
                "gpu_query_retries_total": self.query_retry_count,
                "host": {
                    "cpu_utilization_percent": psutil.cpu_percent(),
                    "cpu_utilization_per_core_percent": psutil.cpu_percent(percpu=True),
                    "cpu_frequency_mhz": frequency.current if frequency else None,
                    "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
                    "memory_used_mib": memory.used / (1024 * 1024),
                    "memory_total_mib": memory.total / (1024 * 1024),
                    "swap_used_mib": swap_used_mib,
                    "disk_io": disk_io._asdict() if disk_io else {},
                    "network_io": psutil.net_io_counters()._asdict(),
                },
            }
            self.samples.append(sample)
            if self.artifact_path is not None:
                with self.artifact_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    handle.flush()
            if self.stdout:
                sys.stdout.write(
                    json.dumps({"event": "telemetry", **sample}, sort_keys=True) + "\n"
                )
                sys.stdout.flush()
                if self.diagnostic:
                    print(
                        json.dumps(
                            {
                                "event": "telemetry_diagnostic",
                                "rank": self.rank,
                                "diagnostic": self.diagnostic,
                            }
                        ),
                        flush=True,
                    )
            self._stop.wait(self.interval_seconds)
