"""Freeze an existing legacy teacher, distill, gate, then run paired old MAPPO."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
from edge_sim_learning.experiment import evaluate
from edge_sim_learning.legacy_distill import distilled_policy, fit
from edge_sim_learning.pretrain import collect, sha256
from edge_sim_learning.scenario import ScenarioConfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="online")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        print("Already complete", flush=True)
        return
    teacher = out / "teacher.pt"
    if not teacher.exists():
        shutil.copy2(args.teacher, out / "teacher.tmp")
        (out / "teacher.tmp").replace(teacher)
    payload = torch.load(teacher, map_location="cpu", weights_only=False)
    cfg = ScenarioConfig.model_validate(payload["scenario"])
    if (
        payload["method"] != "MAPPO-no-context"
        or cfg.max_retries
        or cfg.reward_mode != "business"
        or cfg.episode_loads
        or cfg.scheduler_release != "window"
    ):
        raise ValueError("requires legacy MAPPO, window scheduling, business reward, no retries")
    count, val_count, epochs = (2, 1, 2) if args.smoke else (128, 32, 40)
    eval_count = 1 if args.smoke else 10
    workers, batch = (1, cfg.cycles * 2) if args.smoke else (8, 1024)
    episodes = 4 if args.smoke else 2048
    # Use a different seed from teacher training, identically in both RL arms.
    paired_seed = 17 if payload["seed"] != 17 else 0
    spec = dict(
        teacher_sha256=sha256(teacher),
        teacher_frames=payload["experiment"]["state"]["total_frames"],
        teacher_training_seed=payload["seed"],
        paired_training_seed=paired_seed,
        scenario=cfg.model_dump(),
        train_episodes=count,
        supervised_validation_episodes=val_count,
        distillation_epochs=epochs,
        validation_episodes=eval_count,
        rl_episodes=episodes,
        rl_workers=workers,
        rl_batch=batch,
        device=args.device,
        smoke=args.smoke,
        gate="success deficit <= 1 percentage point and mean success latency <= 105% of teacher",
    )
    spec_path = out / "spec.json"
    canonical = json.loads(json.dumps(spec))
    if spec_path.exists() and json.loads(spec_path.read_text()) != canonical:
        raise ValueError("trial configuration changed; use another output")
    spec_path.write_text(json.dumps(canonical, indent=2))
    print("COLLECT", flush=True)
    manifest = collect(
        teacher,
        out / "data",
        train_episodes=count,
        validation_episodes=val_count,
        workers=1 if args.smoke else 4,
    )
    print("DISTILL", flush=True)
    with wandb.init(
        project="ecsai-deppo",
        mode=args.wandb_mode,
        dir=str(out),
        entity=os.environ.get("WANDB_ENTITY"),
        name="legacy-independent-distillation",
        config=canonical,
    ) as run:
        run.define_metric("epoch")
        run.define_metric("distill/*", step_metric="epoch")

        def log(row):
            run.log(
                {"epoch": row["epoch"]}
                | {"distill/" + k: v for k, v in row.items() if k != "epoch" and v is not None}
            )

        student = fit(
            manifest,
            out / "student",
            epochs=epochs,
            batch_size=128,
            resume=(out / "student/last.pt").exists(),
            on_epoch=log,
        )
        print("VALIDATE FROZEN TEACHER AND STUDENT", flush=True)
        reports = {}
        seeds = list(range(1_000_000_000, 1_000_000_000 + eval_count))
        for name, policy in [
            ("teacher", payload["policy"]),
            ("student", distilled_policy(payload["policy"], student)),
        ]:
            path = out / (name + "-evaluation.json")
            # The trial spec and deterministic fitting/resume fix these artifacts.
            result = (
                json.loads(path.read_text())
                if path.exists()
                else evaluate(
                    cfg,
                    "MAPPO-no-context",
                    policy,
                    seeds,
                    exploration="stochastic",
                    workers=1 if args.smoke else 4,
                )
            )
            path.write_text(json.dumps(result, indent=2))
            reports[name] = result
            run.log({name + "/" + k: v for k, v in result["mean"].items()})
        a, b = [reports[k]["mean"] for k in ["teacher", "student"]]
        gap = b["success_rate"] - a["success_rate"]
        latency_pass = b["mean_latency_s"] <= 1.05 * a["mean_latency_s"]
        gate = dict(
            passed=gap >= -0.01 and latency_pass,
            success_gap_pp=100 * gap,
            teacher_latency_s=a["mean_latency_s"],
            student_latency_s=b["mean_latency_s"],
            validation_only=True,
        )
        (out / "gate.json").write_text(json.dumps(gate, indent=2))
        run.summary.update(gate)
    if not gate["passed"]:
        print("STUDENT NOT READY: no RL launched", gate, flush=True)
        return
    print("PAIRED RL", flush=True)
    scenario = out / "scenario.json"
    scenario.write_text(cfg.model_dump_json(indent=2))

    def run_arm(arm):
        dest = out / arm
        cmd = [
            sys.executable,
            "-m",
            "edge_sim_learning.cli",
            "train",
            "--scenario",
            str(scenario),
            "--output",
            str(dest),
            "--method",
            "MAPPO-no-context",
            "--seed",
            str(paired_seed),
            "--episodes",
            str(episodes),
            "--workers",
            str(workers),
            "--batch",
            str(batch),
            "--minibatch",
            str(batch),
            "--epochs",
            "5",
            "--learning-rate",
            "0.0003",
            "--normalize-advantage",
            "--hidden-size",
            "256",
            "--context-size",
            "128",
            "--initial-std",
            "0.3",
            "--device",
            args.device,
            "--wandb-mode",
            args.wandb_mode,
            "--eval-interval",
            str(batch if args.smoke else 8192),
            "--eval-episodes",
            str(eval_count),
            "--eval-workers",
            str(1 if args.smoke else 4),
            "--eval-stochastic",
        ]
        if (dest / "last.pt").exists():
            state = torch.load(dest / "last.pt", map_location="cpu", weights_only=False)
            if state["experiment"]["state"]["total_frames"] >= episodes * cfg.cycles:
                return arm
            cmd += ["--resume", str(dest / "last.pt")]
        elif arm == "pretrained":
            cmd += ["--actor-init", str(student)]
        with (out / (arm + ".log")).open("a") as stream:
            child = subprocess.Popen(
                cmd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    BC_EVAL_LOCK=str(out / "evaluation.lock"),
                    ECSAI_RUN_NAME=f"MAPPO-legacy-{arm}-seed{paired_seed}",
                ),
            )
            (out / (arm + "-process.json")).write_text(json.dumps(dict(pid=child.pid, args=cmd)))
            if child.wait():
                raise RuntimeError(f"{arm} failed; last checkpoint retained")
        return arm

    with ThreadPoolExecutor(max_workers=2) as pool:
        finished = list(pool.map(run_arm, ["scratch", "pretrained"]))
    initial = [
        torch.load(out / arm / "initial.pt", map_location="cpu", weights_only=False)
        for arm in finished
    ]
    a, b = [p["experiment"]["loss_agents"] for p in initial]
    keys = [k for k in a if "critic" in k and torch.is_tensor(a[k])]
    if not keys or not all(torch.equal(a[k], b[k]) for k in keys):
        raise ValueError("paired initial critics differ")
    for arm in finished:
        path = out / arm / f"evaluation-stochastic-{episodes * cfg.cycles}.json"
        reports[arm] = json.loads(path.read_text())
    workloads = [[(e["seed"], e["workload"]) for e in r["episodes"]] for r in reports.values()]
    if not all(w == workloads[0] for w in workloads):
        raise ValueError("paired evaluation workloads differ")
    rows = [dict(method=name, **r["mean"]) for name, r in reports.items()]
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    with (out / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                runs=finished,
                performance_claim=False,
                paired_critics=True,
                paired_workloads=True,
            )
        )
    )


if __name__ == "__main__":
    main()
