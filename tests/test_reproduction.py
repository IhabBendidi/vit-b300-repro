import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from vit_benchmark.config import BenchmarkConfig, load_config
from vit_benchmark.imaging import SyntheticImageSpec, SyntheticImagingDataset, execute_imaging
from vit_benchmark.imaging_zarr import (
    ExportPlan,
    checksum_store,
    configure_worker,
    create_store,
    make_plan,
    verify_samples,
    write_shard,
)
from vit_benchmark.zarr_stream import decode_shard, direct_read, iter_shards
from vit_benchmark.zarr_training import verify_local_dataset

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch", [1024, 2048, 3072])
def test_full_config_preserves_architecture_and_eight_gpu_batch(batch):
    config = load_config(ROOT / "config" / f"b300-8gpu-b{batch}.toml")
    s = config.imaging
    assert (
        s.image_size,
        s.patch_size,
        s.embedding_size,
        s.transformer_depth,
        s.attention_heads,
        s.mlp_size,
    ) == (224, 16, 768, 12, 12, 3072)
    assert s.expected_world_size == config.provider.gpu_count == 8
    assert s.micro_batch_size_per_rank == batch
    assert s.global_batch_size == batch * 8
    assert s.gradient_accumulation_steps(8) == 1
    assert config.run.precision == "bf16"


def test_generator_is_deterministic_with_disjoint_compounds():
    train = SyntheticImagingDataset(SyntheticImageSpec(7, 80, 16, 12, tuple(range(80))))
    validation = SyntheticImagingDataset(SyntheticImageSpec(7, 20, 16, 12, tuple(range(80, 100))))
    x, y = train[3]
    assert torch.equal(x, train[3][0]) and torch.equal(y, train[3][1])
    assert x.shape == (3, 16, 16) and x.dtype == torch.float32
    assert set(train.spec.compound_ids).isdisjoint(validation.spec.compound_ids)
    assert {int(validation[i][1]) for i in range(20)} == set(range(12))


def test_real_cpu_training_writes_complete_record(tmp_path, capsys):
    torch.set_num_threads(1)
    config = BenchmarkConfig.model_validate(
        {
            "run": {
                "run_id": "cpu-check",
                "mode": "gpu_only",
                "precision": "fp32",
                "output_dir": str(tmp_path / "results"),
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "provider": {
                "name": "local",
                "region": "local",
                "instance_type": "cpu",
                "gpu_model": "none",
                "gpu_count": 1,
                "node_count": 1,
                "network_description": "none",
                "storage_description": "temporary files",
            },
            "imaging": {
                "dataset_size": 24,
                "validation_sample_count": 6,
                "class_count": 12,
                "compound_count": 100,
                "train_compound_count": 80,
                "image_size": 16,
                "patch_size": 8,
                "embedding_size": 32,
                "transformer_depth": 1,
                "attention_heads": 4,
                "mlp_size": 64,
                "global_batch_size": 4,
                "micro_batch_size_per_rank": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            },
            "timing": {
                "warmup_seconds": 0.001,
                "measurement_seconds": 0.001,
                "measurement_repetitions": 1,
                "fixed_job_updates": 2,
            },
        }
    )
    path = execute_imaging(config, "test")
    record = json.loads(path.read_text())
    assert record["fixed_job"]["samples_processed"] == 8
    assert record["fixed_job"]["optimizer_updates"] == 2
    assert record["fixed_job"]["validation_loss"] > 0
    assert record["gpu_telemetry"]["sample_count"] > 0
    assert record["trials"][0]["throughput_samples_per_second"] > 0
    state = torch.load(tmp_path / "artifacts/cpu-check.pt", weights_only=True)
    assert "patch_embedding.weight" in state and "head.weight" in state
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "run_complete"


def test_full_dataset_plan_is_exact_and_bounded():
    p = make_plan(224, 140_000_000_000, 150_000_000_000)
    assert (p.train_count, p.validation_count, p.chunk_images, p.shard_images) == (
        185856,
        46592,
        8,
        256,
    )
    assert p.logical_bytes == 232448 * (602112 + 10)
    with pytest.raises(ValueError):
        make_plan(224, 150_000_000_000, 150_000_000_000)


def test_zarr_roundtrip_decode_integrity_and_overwrite_protection(tmp_path):
    configure_worker()
    p = ExportPlan(8, 32, 16, 8, 16, 48 * (768 + 10), 2_000_000_000)
    path = tmp_path / "tiny.zarr"
    create_store(path, p, {"plan": asdict(p)})
    for split, count, compounds in [
        ("train", 32, tuple(range(80))),
        ("validation", 16, tuple(range(80, 100))),
    ]:
        for start in range(0, count, 16):
            write_shard((str(path), split, start, 16, 8, 123, count, compounds))
    root = zarr.open_group(str(path), mode="r")
    assert verify_samples(path, p, 123) > 10
    metadata = (path / "train/images/zarr.json").read_bytes()
    payload = (path / "train/images/c/0/0/0/0").read_bytes()
    np.testing.assert_array_equal(decode_shard(metadata, payload, 0), root["train/images"][:16])
    corrupt = bytearray(payload)
    corrupt[0] ^= 1
    with pytest.raises(Exception):
        decode_shard(metadata, bytes(corrupt), 0)
    assert checksum_store(path) > 0
    for line in (path / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        create_store(path, p, {})


@pytest.mark.parametrize("shards_per_batch", [4, 8, 12])
def test_shard_assignment_is_disjoint_balanced_and_rotates(shards_per_batch):
    ranks = [list(iter_shards(726, rank, 8, 0, 7, shards_per_batch)) for rank in range(8)]
    all_shards = [item for rank in ranks for item in rank]
    assert len(all_shards) == len(set(all_shards))
    assert len({len(rank) for rank in ranks}) == 1
    assert all(len(rank) % shards_per_batch == 0 for rank in ranks)
    assert ranks[0] != list(iter_shards(726, 0, 8, 1, 7, shards_per_batch))


def dataset_receipt(tmp_path):
    manifest = {
        "status": "complete",
        "generator": "synthetic_imaging_v1",
        "seed": 20261005,
        "class_count": 12,
        "layout": "NCHW",
        "dtype": "float32",
        "train_compounds": list(range(80)),
        "validation_compounds": list(range(80, 100)),
        "plan": asdict(make_plan(224, 140_000_000_000, 150_000_000_000)),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    payload = tmp_path / "payload"
    payload.write_bytes(b"checksum test")
    (tmp_path / "checksums.sha256").write_text(
        hashlib.sha256(payload.read_bytes()).hexdigest() + "  payload\n"
    )
    (tmp_path / "_SUCCESS").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(
                    (tmp_path / "manifest.json").read_bytes()
                ).hexdigest(),
                "checksums_sha256": hashlib.sha256(
                    (tmp_path / "checksums.sha256").read_bytes()
                ).hexdigest(),
            }
        )
    )


def test_receipts_and_full_hash_verifier_detect_tampering(tmp_path):
    dataset_receipt(tmp_path)
    verifier = load_script("verify_dataset")
    assert verifier.verify(tmp_path, 20261005)["files"] == 1
    (tmp_path / "payload").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verifier.verify(tmp_path, 20261005)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksums differ"):
        verify_local_dataset(tmp_path, 20261005)


def test_direct_io_is_real_or_explicitly_unavailable(tmp_path):
    path = tmp_path / "aligned"
    path.write_bytes(b"x" * 8192)
    if not sys.platform.startswith("linux"):
        with pytest.raises(RuntimeError, match="requires Linux"):
            direct_read(path)
        return
    blob, stats = direct_read(path)
    assert bytes(blob) == b"x" * 8192
    assert stats["physical_bytes"] >= 8192


def test_speed_summary_excludes_warmup_and_uses_correct_gpu_denominator(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "accumulation",
                "gpus",
                "effective_global_batch",
                "images_per_second",
                "image_read_gb_s",
            ],
        )
        writer.writeheader()
        for phase, rate in [
            ("warmup", 999999),
            ("trial-0", 800),
            ("trial-1", 1600),
            ("trial-2", 1200),
        ]:
            writer.writerow(
                {
                    "phase": phase,
                    "accumulation": 1,
                    "gpus": 8,
                    "effective_global_batch": 8192,
                    "images_per_second": rate,
                    "image_read_gb_s": rate * 602112 / 1e9,
                }
            )
    row = load_script("summarize").summarize(path)[0]
    assert row["trials"] == 3 and row["images_s_node"] == 1200
    assert row["images_s_gpu"] == 150 and row["batch_per_gpu"] == 1024
