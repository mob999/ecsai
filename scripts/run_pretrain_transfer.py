"""Source-only BC pretraining followed by matched target-scale adaptation experiments."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from edge_sim_learning.pretrain import load_base_policy, sha256
from edge_sim_learning.scenario import ScenarioConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import torch

    source = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if source["method"] != "MAPPO-no-context" or source.get("action_dim") != 5:
        raise ValueError("requires a five-action no-context teacher")
    target = ScenarioConfig.profile("smoke" if args.smoke else "medium").model_copy(
        update={
            "clusters": 3 if args.smoke else 5,
            "caches": 3 if args.smoke else 20,
            "request_rate": 30 if args.smoke else 600,
            "scheduler_release": "window",
            "delivery_load": 0.75,
        }
    )
    if source["scenario"]["clusters"] == target.clusters:
        raise ValueError("source and target must have different cluster counts")
    steps, batch, eval_count, test_count = (32, 16, 2, 2) if args.smoke else (65536, 512, 10, 30)
    if batch % (args.workers * target.cycles):
        raise ValueError("worker count must divide whole episodes per collection batch")
    train_count, valid_count = (3, 2) if args.smoke else (128, 32)
    bc_epochs, bc_batches = (2, 8) if args.smoke else (20, 128)
    hidden_size = 32 if args.smoke else 256
    spec = dict(
        teacher_sha256=sha256(args.teacher),
        target=target.model_dump(),
        steps=steps,
        batch=batch,
        workers=args.workers,
        hidden_size=hidden_size,
        train_count=train_count,
        valid_count=valid_count,
        bc_epochs=bc_epochs,
        bc_batches=bc_batches,
        seed=0,
        eval_count=eval_count,
        test_count=test_count,
        device=args.device,
    )
    record = root / "suite.json"
    if record.exists() and json.loads(record.read_text()) != spec:
        raise ValueError("output belongs to another transfer experiment")
    record.write_text(json.dumps(spec, indent=2))
    teacher = root / "teacher.pt"
    if not teacher.exists():
        shutil.copy2(args.teacher, teacher)
    if sha256(teacher) != spec["teacher_sha256"]:
        raise ValueError("teacher snapshot changed")
    scenario = root / "target.json"
    scenario.write_text(target.model_dump_json(indent=2))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_RUN_GROUP="bc-transfer-smoke" if args.smoke else "bc-small-to-medium-v1",
    )

    def run(label, command):
        folder = root / label
        folder.mkdir(exist_ok=True)
        print(f"START {label}", flush=True)
        with (folder / "console.log").open("a") as log:
            subprocess.run(
                [sys.executable, *map(str, command)],
                check=True,
                env=env | {"ECSAI_RUN_NAME": label},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"DONE {label}", flush=True)

    run(
        "data",
        [
            "-m",
            "edge_sim_learning.pretrain",
            "collect",
            "--teacher",
            teacher,
            "--output",
            root / "data",
            "--train-episodes",
            train_count,
            "--validation-episodes",
            valid_count,
            "--workers",
            args.workers,
        ],
    )
    base, last = root / "base/best.pt", root / "base/last.pt"
    if not last.exists() or torch.load(last, weights_only=True)["epoch"] < bc_epochs:
        run(
            "base",
            [
                "-m",
                "edge_sim_learning.pretrain",
                "fit",
                "--manifests",
                root / "data/manifest.json",
                "--output",
                root / "base",
                "--epochs",
                bc_epochs,
                "--batches-per-epoch",
                bc_batches,
                "--hidden-size",
                hidden_size,
                "--device",
                args.device,
                "--mode",
                args.wandb_mode,
                *(["--resume"] if last.exists() else []),
            ],
        )
    if not base.exists():
        raise ValueError("BC did not save its best validation checkpoint")
    bc_folder = root / "BC-only"
    bc_folder.mkdir(exist_ok=True)
    if not (bc_folder / "evaluation.json").exists():
        from edge_sim_learning.experiment import evaluate, save_report

        result = evaluate(
            target,
            "MAPPO-no-context",
            load_base_policy(base),
            range(2_000_000_000, 2_000_000_000 + test_count),
            exploration="stochastic",
            workers=args.workers,
        )
        (bc_folder / "evaluation.json").write_text(json.dumps(result, indent=2))
        os.environ["WANDB_RUN_GROUP"] = env["WANDB_RUN_GROUP"]
        os.environ["ECSAI_RUN_NAME"] = "BC-only"
        save_report(
            bc_folder,
            target,
            0,
            "BC-only",
            args.wandb_mode,
            [{"env_steps": 0, **{"test/" + k: v for k, v in result["mean"].items()}}],
        )
    # Exercise the requested pretrained-to-online path first, then matched controls.
    for label in ("BC-full", "Scratch", "BC-head"):
        folder = root / label
        checkpoint = folder / "last.pt"
        complete = (
            checkpoint.exists()
            and torch.load(checkpoint, map_location="cpu", weights_only=False)["experiment"][
                "state"
            ]["total_frames"]
            >= steps
        )
        if not complete:
            extra = (
                ["--resume", checkpoint]
                if checkpoint.exists()
                else (
                    ["--actor-init", base, *(["--head-only"] if label == "BC-head" else [])]
                    if label != "Scratch"
                    else []
                )
            )
            run(
                label,
                [
                    "-m",
                    "edge_sim_learning.cli",
                    "train",
                    "--scenario",
                    scenario,
                    "--output",
                    folder,
                    "--method",
                    "MAPPO-no-context",
                    "--wandb-mode",
                    args.wandb_mode,
                    "--episodes",
                    steps // target.cycles,
                    "--workers",
                    args.workers,
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
                    hidden_size,
                    "--initial-std",
                    0.3,
                    "--device",
                    args.device,
                    "--eval-interval",
                    batch if args.smoke else 8192,
                    "--eval-episodes",
                    eval_count,
                    "--eval-workers",
                    args.workers,
                    "--eval-stochastic",
                    *extra,
                ],
            )
        selected = folder / "best-stochastic.pt"
        if not (root / f"{label}-test/evaluation.json").exists():
            run(
                f"{label}-test",
                [
                    "-m",
                    "edge_sim_learning.cli",
                    "evaluate",
                    "--checkpoint",
                    selected,
                    "--output",
                    root / f"{label}-test",
                    "--wandb-mode",
                    args.wandb_mode,
                    "--episodes",
                    test_count,
                    "--workers",
                    args.workers,
                    "--split",
                    "test",
                    "--exploration",
                    "stochastic",
                ],
            )
    rows, workloads = [], None
    for label in ("BC-only", "Scratch", "BC-full", "BC-head"):
        folder = root / (label if label == "BC-only" else f"{label}-test")
        result = json.loads((folder / "evaluation.json").read_text())
        current = [(r["seed"], r["workload"], r["arrived"]) for r in result["episodes"]]
        if workloads is None:
            workloads = current
        if current != workloads or any(r["unfinished"] for r in result["episodes"]):
            raise ValueError("test workload or drain mismatch")
        rows.append(
            {"method": label, "online_steps": 0 if label == "BC-only" else steps, **result["mean"]}
        )
    (root / "comparison.json").write_text(json.dumps(rows, indent=2))
    data_manifest = json.loads((root / "data/manifest.json").read_text())
    bc_last = torch.load(last, map_location="cpu", weights_only=True)
    costs = {
        "teacher_env_steps": data_manifest["spec"]["teacher_frames"],
        "source_collection_env_steps": sum(r["steps"] for r in data_manifest["episodes"]),
        "source_collection_wall_s": data_manifest.get("collection_wall_s"),
        "source_episode_wall_s_sum": sum(r["wall_s"] for r in data_manifest["episodes"]),
        "bc_updates": bc_last["metrics"]["updates"],
        "bc_wall_s": bc_last["metrics"]["wall_s"],
        "target_env_steps_per_rl_method": steps,
        "note": (
            "Teacher training and source collection are additional costs; "
            "validation/test sampling is separate."
        ),
    }
    (root / "costs.json").write_text(json.dumps(costs, indent=2))
    with (root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "complete.json").write_text(json.dumps({"complete": True, "methods": 4}))
    print("TRANSFER SUITE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
