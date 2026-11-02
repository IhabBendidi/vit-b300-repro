"""Print the software/GPU environment without starting a training job."""

import argparse
import importlib.metadata
import json
import platform
import sys

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-8-gpus", action="store_true")
args = parser.parse_args()
report = {
    "platform": platform.platform(),
    "python": sys.version,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "zarr", "psutil", "pydantic")
    },
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
        }
        for i in range(torch.cuda.device_count())
    ],
}
print(json.dumps(report, indent=2))
if args.require_8_gpus:
    if not sys.platform.startswith("linux"):
        raise SystemExit("The eight-GPU direct-I/O reproduction requires Linux.")
    if torch.cuda.device_count() != 8:
        raise SystemExit("Expected exactly eight visible CUDA GPUs.")
    for index in range(8):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"GPU {index} does not support BF16.")
