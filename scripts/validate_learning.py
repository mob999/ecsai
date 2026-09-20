"""Reproduce the three-seed small-budget comparison, not paper-level training.

Run after `uv sync --all-packages --locked` and native SimGrid installation.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("outputs/deppo-validation"))
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from edge_sim_learning.scenario import ScenarioConfig

    config = ScenarioConfig.profile("small").model_copy(update={"cycles": 16})
    scenario = output / "scenario.json"
    scenario.write_text(config.model_dump_json(indent=2))

    def run(command, folder, extra):
        folder.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            "-m",
            "edge_sim_learning.cli",
            command,
            "--scenario",
            str(scenario),
            "--output",
            str(folder),
            "--wandb-mode",
            "offline",
            *extra,
        ]
        print(" ".join(argv), flush=True)
        with (folder / "console.log").open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)

    rows = []
    workloads = []
    for method in ("DEPPO-adapted", "MAPPO-no-context"):
        for seed in (0, 1, 2):
            folder = output / f"{method}-seed{seed}"
            run(
                "train",
                folder,
                [
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--episodes",
                    "8",
                    "--workers",
                    "2",
                    "--batch",
                    "64",
                    "--epochs",
                    "10",
                    "--minibatch",
                    "64",
                    "--eval-interval",
                    "128",
                    "--eval-episodes",
                    "10",
                ],
            )
            result = json.loads((folder / "evaluation-128.json").read_text())
            workloads.append([episode["workload"] for episode in result["episodes"]])
            rows.append({"method": method, "training_seed": seed, **result["mean"]})
    for method in ("random", "local", "forward"):
        folder = output / method
        run("evaluate", folder, ["--method", method, "--episodes", "10"])
        result = json.loads((folder / "evaluation.json").read_text())
        workloads.append([episode["workload"] for episode in result["episodes"]])
        rows.append({"method": method, "training_seed": None, **result["mean"]})
    run("benchmark", output / "benchmark-small", ["--workers", "1", "2", "4", "--steps", "64"])
    assert all(workload == workloads[0] for workload in workloads), "evaluation workload mismatch"
    (output / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (output / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
