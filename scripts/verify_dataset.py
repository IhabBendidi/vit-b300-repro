"""Verify a completed dataset's receipt and every listed file hash."""

import argparse
import hashlib
import json
from pathlib import Path

from vit_benchmark.zarr_training import verify_local_dataset


def verify(root: Path, seed: int) -> dict:
    root = root.resolve()
    identity = verify_local_dataset(root, seed)
    total = 0
    count = 0
    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root or not path.is_file():
            raise ValueError(f"Invalid dataset file: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {name}")
        total += path.stat().st_size
        count += 1
    if not count:
        raise ValueError("The checksum list is empty")
    return {"status": "verified", "files": count, "bytes": total, **identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.seed), indent=2))
