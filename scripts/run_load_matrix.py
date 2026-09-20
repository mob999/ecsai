"""Matched three-scale baseline/DD/MAPPO suite; run locally with --smoke first."""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge_sim_learning.scenario import ScenarioConfig

BASELINES = [
    ("Random", "random", None),
    ("Always-local", "local", None),
    ("Always-forward", "forward", None),
    ("Forward-r0.6", "forward", 0.6),
    ("Queue-adaptive", "queue-adaptive", None),
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    steps, batch, count = (32, 16, 2) if args.smoke else (262144, 1024, 10)
    workers = 2 if args.smoke else args.workers
    configs = {}
    for name, clusters, caches, rate in [
        ("small", 3, 10, 300),
        ("medium", 5, 20, 600),
        ("large", 7, 30, 900),
    ]:
        config = ScenarioConfig.profile("smoke" if args.smoke else "small")
        dimensions = {"clusters": clusters, "caches": caches, "request_rate": rate}
        if args.smoke:
            dimensions = {
                "clusters": {"small": 2, "medium": 3, "large": 4}[name],
                "caches": {"small": 2, "medium": 4, "large": 6}[name],
                "request_rate": 30,
            }
        configs[f"scale-{name}"] = config.model_copy(
            update={"scheduler_release": "window", "delivery_load": 0.75, **dimensions}
        )
    spec = {
        "scenarios": {k: v.model_dump() for k, v in configs.items()},
        "steps": steps,
        "batch": batch,
        "eval_episodes": count,
        "workers": workers,
        "training_seeds": [0],
        "parallel": args.parallel,
        "device": args.device,
        "wandb_mode": args.wandb_mode,
        "version": 3,
    }
    manifest = root / "suite.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Existing output has a different experiment specification")
    manifest.write_text(json.dumps(spec, indent=2))

    def job(load, label, command, extra):
        cfg = configs[load]
        folder = root / load / label
        folder.mkdir(parents=True, exist_ok=True)
        scenario = folder / "scenario.json"
        scenario.write_text(cfg.model_dump_json(indent=2))
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
            args.wandb_mode,
            *map(str, extra),
        ]
        signature = {"argv": argv, "scenario": cfg.model_dump()}
        record = folder / "job.json"
        if record.exists() and json.loads(record.read_text()) != signature:
            raise ValueError(f"Changed job: {folder}")
        record.write_text(json.dumps(signature, indent=2))
        expected = folder / (
            f"evaluation-stochastic-{steps}.json" if command == "train" else "evaluation.json"
        )
        if (folder / "complete.json").exists():
            if not expected.exists():
                raise ValueError(f"Missing completed evaluation: {folder}")
            return
        if command == "train" and (folder / "last.pt").exists():
            import torch

            payload = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            if payload["experiment"]["state"]["total_frames"] >= steps:
                if not expected.exists():
                    raise ValueError(f"Training complete but final evaluation missing: {folder}")
                (folder / "complete.json").write_text('{"complete": true}')
                return
            argv += ["--resume", str(folder / "last.pt")]
            del payload
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            WANDB_RUN_GROUP=f"window-{load}",
            ECSAI_RUN_NAME=f"{label}-{load}-seed0",
        )
        print(f"START {load}/{label}", flush=True)
        with (folder / "console.log").open("a") as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not expected.exists():
            raise ValueError(f"Missing evaluation: {folder}")
        (folder / "complete.json").write_text('{"complete": true}')
        print(f"DONE {load}/{label}", flush=True)

    # Finish inexpensive references first, then keep two independent learners running.
    for load in configs:
        for label, method, ratio in BASELINES:
            extra = [
                "--method",
                method,
                "--episodes",
                count,
                "--workers",
                2 if args.smoke else 4,
                "--split",
                "validation",
                "--seed",
                0,
            ]
            if ratio is not None:
                extra += ["--fixed-ratio", ratio]
            job(load, label, "evaluate", extra)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = []
        for load, cfg in configs.items():
            for label, method in [("MAPPO", "MAPPO-no-context"), ("DD", "DD-adapted")]:
                extra = [
                    "--method",
                    method,
                    "--seed",
                    0,
                    "--episodes",
                    steps // cfg.cycles,
                    "--workers",
                    workers,
                    "--batch",
                    batch,
                    "--minibatch",
                    batch,
                    "--epochs",
                    5,
                    "--learning-rate",
                    0.0003,
                    "--normalize-advantage",
                    "--hidden-size",
                    256,
                    "--context-size",
                    128,
                    "--initial-std",
                    0.3,
                    "--device",
                    args.device,
                    "--eval-interval",
                    batch if args.smoke else 8192,
                    "--eval-episodes",
                    count,
                    "--eval-workers",
                    2 if args.smoke else 4,
                    "--eval-stochastic",
                ]
                futures.append(pool.submit(job, load, label, "train", extra))
        for future in futures:
            future.result()
    rows = []
    for load, cfg in configs.items():
        reference = None
        for label in [x[0] for x in BASELINES] + ["MAPPO", "DD"]:
            folder = root / load / label
            name = (
                f"evaluation-stochastic-{steps}.json"
                if label in {"MAPPO", "DD"}
                else "evaluation.json"
            )
            result = json.loads((folder / name).read_text())
            assert json.loads((folder / "config.json").read_text())["scenario"] == cfg.model_dump()
            workloads = [(e["seed"], e["arrived"], e["workload"]) for e in result["episodes"]]
            if reference is None:
                reference = workloads
            assert workloads == reference, f"Workload mismatch: {load}/{label}"
            assert all(
                e["unfinished"] == 0
                and e["arrived"] == e["completed"] + e["timed_out"] + e["rejected"]
                for e in result["episodes"]
            )
            rows.append({"load": load, "method": label, **result["mean"]})
    (root / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "complete.json").write_text(json.dumps({"complete": True, "verified_jobs": len(rows)}))
    print("SUITE COMPLETE: matched workloads, configs and request accounting verified", flush=True)


if __name__ == "__main__":
    main()
