"""Print measured throughput from one completed local training run."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(path: Path) -> list[dict]:
    if path.suffix == ".json":
        result = json.loads(path.read_text())
        rates = [t["throughput_samples_per_second"] for t in result["trials"]]
        return [
            {
                "mode": "gpu_only",
                "gpus": result["world_size"],
                "batch_per_gpu": result["micro_batch_size_per_rank"],
                "accumulation": result["gradient_accumulation_steps"],
                "trials": len(rates),
                "images_s_node": statistics.median(rates),
                "images_s_gpu": statistics.median(rates) / result["world_size"],
                "trial_min_images_s_node": min(rates),
                "trial_max_images_s_node": max(rates),
            }
        ]
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"].startswith("trial-")]
    summaries = []
    for accumulation in sorted({int(row["accumulation"]) for row in rows}):
        group = [row for row in rows if int(row["accumulation"]) == accumulation]
        gpus = int(group[0]["gpus"])
        rates = [float(row["images_per_second"]) for row in group]
        summaries.append(
            {
                "mode": "disk_fed",
                "gpus": gpus,
                "batch_per_gpu": int(group[0]["effective_global_batch"]) // (gpus * accumulation),
                "accumulation": accumulation,
                "trials": len(rates),
                "images_s_node": statistics.median(rates),
                "images_s_gpu": statistics.median(rates) / gpus,
                "trial_min_images_s_node": min(rates),
                "trial_max_images_s_node": max(rates),
                "effective_read_GB_s_node": statistics.median(
                    float(row["image_read_gb_s"]) for row in group
                ),
            }
        )
    if not summaries:
        raise ValueError("No timed trial rows found")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.result), indent=2))
