"""Run the predeclared validation search and held-out comparison, resumably.

Local: .venv/bin/python scripts/run_deppo_v2.py --smoke --output outputs/v2-smoke
GPU:   .venv/bin/python scripts/run_deppo_v2.py --device cuda --output outputs/v2-small
Completed jobs are skipped. Failed training stops the suite for diagnosis.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

METHODS = ("DEPPO-adapted", "MAPPO-no-context")


def paired_interval(learned, baseline, seed=42):
    """Hierarchical bootstrap over training seeds and paired evaluation workloads."""
    differences = np.asarray(learned) - np.asarray(baseline)[None, :]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        rows = rng.integers(0, differences.shape[0], differences.shape[0])
        cols = rng.integers(0, differences.shape[1], differences.shape[1])
        samples.append(differences[np.ix_(rows, cols)].mean())
    return {
        "difference": float(differences.mean()),
        "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--profile", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--load", type=float, choices=[0.55, 0.75, 0.95], default=0.75)
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from edge_sim_learning.scenario import ScenarioConfig

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = ScenarioConfig.profile("smoke" if args.smoke else args.profile).model_copy(
        update={"delivery_load": args.load}
    )
    batch, workers = (16, 1) if args.smoke else (512, 4)
    search_steps, final_steps = (32, 64) if args.smoke else (65536, 262144)
    val_count, test_count = (2, 3) if args.smoke else (10, 30)
    interval = batch if args.smoke else 8192
    spec = dict(
        scenario=config.model_dump(),
        search_steps=search_steps,
        final_steps=final_steps,
        seeds=[0, 1, 2],
        val_count=val_count,
        test_count=test_count,
        device=args.device,
        wandb_mode=args.wandb_mode,
        format_version=2,
    )
    manifest = root / "suite.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("output belongs to a different experiment; use a new directory")
    manifest.write_text(json.dumps(spec, indent=2))

    def job(name, command, cfg, extra):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        signature = dict(command=command, scenario=cfg.model_dump(), args=extra)
        record = folder / "job.json"
        if record.exists() and json.loads(record.read_text()) != signature:
            raise ValueError(f"job configuration changed: {name}")
        record.write_text(json.dumps(signature, indent=2))
        if (folder / "complete.json").exists():
            return folder
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
        # Interrupted training resumes only from an existing fully saved checkpoint.
        if command == "train" and (folder / "last.pt").exists():
            import torch

            last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            completed = last["experiment"]["state"]["total_frames"]
            budget = int(extra[extra.index("--episodes") + 1]) * cfg.cycles
            if completed >= budget:
                (folder / "complete.json").write_text('{"completed": true}')
                return folder
            argv += ["--resume", str(folder / "last.pt")]
            del last
        print(f"Running {name}", flush=True)
        with (folder / "console.log").open("a") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        (folder / "complete.json").write_text('{"completed": true}')
        return folder

    def training(name, cfg, method, seed, steps, lr):
        return job(
            name,
            "train",
            cfg,
            [
                "--method",
                method,
                "--seed",
                seed,
                "--episodes",
                steps // cfg.cycles,
                "--workers",
                workers,
                "--batch",
                batch,
                "--epochs",
                5,
                "--minibatch",
                batch,
                "--learning-rate",
                lr,
                "--device",
                args.device,
                "--eval-interval",
                interval,
                "--eval-episodes",
                val_count,
            ],
        )

    def score(report):
        return report["mean"]["success_rate"], -report["mean"]["mean_latency_s"]

    # Freeze and report baseline allocation tuning using validation workloads only.
    fixed = []
    for rule in ("random", "local", "forward"):
        for n in range(1, 10):
            folder = job(
                f"fixed-search/{rule}-{n}",
                "evaluate",
                config,
                ["--method", rule, "--fixed-ratio", n / 10, "--episodes", val_count],
            )
            report = json.loads((folder / "evaluation.json").read_text())
            fixed.append((score(report), rule, n / 10))
    _, fixed_rule, fixed_ratio = max(fixed, key=lambda x: x[0])
    selections = {"Tuned-fixed": {"rule": fixed_rule, "ratio": fixed_ratio}}
    for method in METHODS:
        trials = []
        for reward in ("business", "paper"):
            cfg = config.model_copy(update={"reward_mode": reward})
            for lr in (1e-4, 3e-4):
                folder = training(
                    f"search/{method}-{reward}-{lr}", cfg, method, 0, search_steps, lr
                )
                reports = [
                    json.loads(p.read_text()) for p in sorted(folder.glob("evaluation-*.json"))
                ]
                trials.append((max(map(score, reports)), reward, lr))
        _, reward, lr = max(trials, key=lambda x: x[0])
        selections[method] = {"reward": reward, "learning_rate": lr, "trials": trials}
    (root / "selection.json").write_text(json.dumps(selections, indent=2))

    # Train from scratch after selection; test is never used for checkpoint selection.
    reports = {}
    for method in METHODS:
        choice = selections[method]
        cfg = config.model_copy(update={"reward_mode": choice["reward"]})
        reports[method] = []
        for seed in (0, 1, 2):
            folder = training(
                f"final/{method}-{seed}", cfg, method, seed, final_steps, choice["learning_rate"]
            )
            result = job(
                f"test/{method}-{seed}",
                "evaluate",
                cfg,
                [
                    "--checkpoint",
                    str(folder / "best.pt"),
                    "--split",
                    "test",
                    "--episodes",
                    test_count,
                ],
            )
            reports[method].append(json.loads((result / "evaluation.json").read_text()))
    for name in ("random", "local", "forward", "queue-adaptive", "Tuned-fixed"):
        extra = [
            "--method",
            fixed_rule if name == "Tuned-fixed" else name,
            "--split",
            "test",
            "--episodes",
            test_count,
        ]
        if name == "Tuned-fixed":
            extra += ["--fixed-ratio", fixed_ratio]
        folder = job(f"test/{name}", "evaluate", config, extra)
        reports[name] = [json.loads((folder / "evaluation.json").read_text())]
    reference = reports["random"][0]["episodes"]
    for group in reports.values():
        for report in group:
            assert [e["workload"] for e in report["episodes"]] == [e["workload"] for e in reference]
    summary = {}
    for name, group in reports.items():
        rates = [r["mean"]["success_rate"] for r in group]
        summary[name] = {
            "success_mean": float(np.mean(rates)),
            "success_seed_std": float(np.std(rates)),
            "training_seeds": len(group) if name in METHODS else 0,
        }
        if name in METHODS:
            values = [[e["success_rate"] for e in r["episodes"]] for r in group]
            summary[name]["paired_comparisons"] = {
                baseline: paired_interval(
                    values, [e["success_rate"] for e in reports[baseline][0]["episodes"]]
                )
                for baseline in ("random", "local", "forward", "queue-adaptive", "Tuned-fixed")
            }
    (root / "comparison.json").write_text(
        json.dumps({"summary": summary, "reports": reports}, indent=2)
    )
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "training_seed", "success_rate", "latency_p95_s", "mean_latency_s"]
        )
        for name, group in reports.items():
            for seed, report in enumerate(group):
                writer.writerow(
                    [
                        name,
                        seed if name in METHODS else "",
                        *[
                            report["mean"][k]
                            for k in ("success_rate", "latency_p95_s", "mean_latency_s")
                        ],
                    ]
                )
    job(
        "benchmark",
        "benchmark",
        config,
        ["--workers", 1, 2, 4, "--steps", 16 if args.smoke else 128],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
