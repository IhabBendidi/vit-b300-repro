"""Atomic JSON output."""

from pathlib import Path

from vit_benchmark.contracts import BenchmarkRunRecord


def write_record(record: BenchmarkRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{record.run_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
