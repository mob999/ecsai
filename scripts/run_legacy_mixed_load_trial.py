"""Overnight matched-data mixed-load distillation and fixed-load RL experiment."""

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
from edge_sim_learning.pretrain import SPLIT_SEEDS, collect, sha256
from edge_sim_learning.scenario import ScenarioConfig

LOADS = (0.75, 0.875, 1.0)


def subset_manifest(source, dest, counts):
    """Reference complete, checksummed episode shards without copying or relabelling them."""
    info = json.loads(source.read_text())
    info["spec"]["counts"] = counts
    info["episodes"] = [
        dict(row, file=str((source.parent / row["file"]).resolve()))
        for row in info["episodes"]
        if row["seed"] < SPLIT_SEEDS[row["split"]] + counts[row["split"]]
    ]
    dest.write_text(json.dumps(info, indent=2))
    return dest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-reference", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        print("Already complete", flush=True)
        return
    torch.set_num_threads(1)
    teacher = out / "teacher.pt"
    if not teacher.exists():
        shutil.copy2(args.teacher, out / "teacher.tmp")
        (out / "teacher.tmp").replace(teacher)
    payload = torch.load(teacher, map_location="cpu", weights_only=False)
    base = ScenarioConfig.model_validate(payload["scenario"])
    assert payload["method"] == "MAPPO-no-context"
    assert base.scheduler_release == "window" and base.reward_mode == "business"
    assert not base.max_retries and not base.episode_loads and base.delivery_load == 0.75
    n, val, epochs = (2, 1, 2) if args.smoke else (128, 32, 40)
    workers, batch = (1, base.cycles * 2) if args.smoke else (8, 1024)
    episodes = 4 if args.smoke else 2048
    eval_n, eval_workers = (1, 1) if args.smoke else (10, 4)
    seeds = list(range(1_000_000_000, 1_000_000_000 + eval_n))
    spec = dict(
        teacher_sha256=sha256(teacher),
        teacher_frames=payload["experiment"]["state"]["total_frames"],
        base=base.model_dump(),
        loads=LOADS,
        mixed_train_per_load=n,
        mixed_val_per_load=val,
        single_train=n * 3,
        single_val=val * 3,
        epochs=epochs,
        seed=0,
        rl_episodes=episodes,
        rl_workers=workers,
        batch=batch,
        eval_seeds=seeds,
        device="cpu" if args.smoke else "cuda",
        scratch_reference=str(args.scratch_reference.resolve()) if args.scratch_reference else None,
        scope="fixed scale, fixed load per episode; mixed offline data only; no retraining teacher",
    )
    spec = json.loads(json.dumps(spec))
    sp = out / "spec.json"
    if sp.exists() and json.loads(sp.read_text()) != spec:
        raise ValueError("changed trial spec; choose another output")
    sp.write_text(json.dumps(spec, indent=2))
    configs, manifests = {}, {}
    for load in LOADS:
        cfg = base.model_copy(
            update={
                "request_rate": base.request_rate * load / base.delivery_load,
                "delivery_load": load,
            }
        )
        configs[load] = cfg
        proxy = out / f"collection-policy-rho{load:g}.pt"
        if not proxy.exists():
            # Policy weights unchanged: scenario override is solely for trajectory collection.
            torch.save(
                payload
                | {"scenario": cfg.model_dump(), "collection_source_sha256": sha256(teacher)},
                proxy,
            )
        print(f"COLLECT rho={load:g}", flush=True)
        manifests[load] = collect(
            proxy,
            out / f"data-rho{load:g}",
            train_episodes=n * 3 if load == 0.75 else n,
            validation_episodes=val * 3 if load == 0.75 else val,
            workers=1 if args.smoke else 4,
        )
    subset = subset_manifest(
        manifests[0.75], out / "mixed075-manifest.json", dict(train=n, validation=val)
    )
    students = {}
    with wandb.init(
        project="ecsai-deppo",
        name="Legacy-mixed-load-pretrain-v1",
        mode=args.wandb_mode,
        dir=str(out),
        config=spec,
    ) as run:
        (out / "wandb.json").write_text(json.dumps({"url": run.url}))
        for name, manifest in [
            ("mixed", [subset, manifests[0.875], manifests[1.0]]),
            ("single", manifests[0.75]),
        ]:
            print("DISTILL", name, flush=True)
            run.define_metric(name + "/epoch")
            run.define_metric(name + "/*", step_metric=name + "/epoch")

            def log(row, name=name):
                run.log({name + "/" + k: v for k, v in row.items() if v is not None})

            students[name] = fit(
                manifest,
                out / name,
                epochs=epochs,
                batch_size=128,
                seed=0,
                resume=(out / name / "last.pt").exists(),
                on_epoch=log,
            )
        gates = []
        frozen = []
        for load, cfg in configs.items():
            results = {}
            for name, policy in [("teacher", payload["policy"])] + [
                (name, distilled_policy(payload["policy"], ck)) for name, ck in students.items()
            ]:
                dest = out / f"frozen-{name}-rho{load:g}.json"
                print("FROZEN EVAL", name, load, flush=True)
                result = (
                    json.loads(dest.read_text())
                    if dest.exists()
                    else evaluate(
                        cfg,
                        "MAPPO-no-context",
                        policy,
                        seeds,
                        exploration="stochastic",
                        workers=eval_workers,
                    )
                )
                dest.write_text(json.dumps(result, indent=2))
                results[name] = result
                frozen.append(dict(method=name, load=load, **result["mean"]))
                run.log({f"frozen/rho{load:g}/{name}/{k}": v for k, v in result["mean"].items()})
            reference = results["teacher"]
            for name in students:
                assert [e["workload"] for e in reference["episodes"]] == [
                    e["workload"] for e in results[name]["episodes"]
                ]
                a, b = reference["mean"], results[name]["mean"]
                passed = (
                    b["success_rate"] >= a["success_rate"] - 0.01
                    and b["mean_latency_s"] <= a["mean_latency_s"] * 1.05
                )
                gates.append(
                    dict(
                        method=name,
                        load=load,
                        passed=passed,
                        required=(name == "mixed" or load == 0.75),
                        success_gap_pp=100 * (b["success_rate"] - a["success_rate"]),
                    )
                )
        (out / "frozen-comparison.json").write_text(json.dumps(frozen, indent=2))
        gate = dict(passed=all(g["passed"] for g in gates if g["required"]), details=gates)
        (out / "gate.json").write_text(json.dumps(gate, indent=2))
        run.summary["gate_passed"] = gate["passed"]
    if not gate["passed"] and not args.smoke:
        (out / "blocked.json").write_text(
            json.dumps({"reason": "frozen student gate failed", **gate})
        )
        print("STUDENT GATE FAILED; preserved results, no RL launched", flush=True)
        return

    def run_arm(job):
        load, arm = job
        cfg = configs[load]
        folder = out / f"rl-rho{load:g}-{arm}"
        scenario = out / f"scenario-rho{load:g}.json"
        # One identical file per load, written before starting concurrent children below.
        reference = args.scratch_reference if load == 0.75 and arm == "scratch" else None
        if reference is not None:
            saved = torch.load(reference / "last.pt", map_location="cpu", weights_only=False)
            if (
                ScenarioConfig.model_validate(saved["scenario"]) != cfg
                or saved["seed"] != 0
                or saved["experiment"]["state"]["total_frames"] != episodes * cfg.cycles
            ):
                raise ValueError("existing scratch reference mismatch")
            return load, arm, reference
        cmd = [
            sys.executable,
            "-m",
            "edge_sim_learning.cli",
            "train",
            "--scenario",
            str(scenario),
            "--output",
            str(folder),
            "--method",
            "MAPPO-no-context",
            "--seed",
            "0",
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
            spec["device"],
            "--wandb-mode",
            args.wandb_mode,
            "--eval-interval",
            str(batch if args.smoke else 8192),
            "--eval-episodes",
            str(eval_n),
            "--eval-workers",
            str(eval_workers),
            "--eval-stochastic",
        ]
        if (folder / "last.pt").exists():
            last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            if last["experiment"]["state"]["total_frames"] >= episodes * cfg.cycles:
                return load, arm, folder
            cmd += ["--resume", str(folder / "last.pt")]
        elif arm in students:
            cmd += ["--actor-init", str(students[arm])]
        print("RL START", load, arm, flush=True)
        with (out / f"rl-rho{load:g}-{arm}.log").open("a") as stream:
            child = subprocess.Popen(
                cmd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    BC_EVAL_LOCK=str(out / "evaluation.lock"),
                    ECSAI_RUN_NAME=f"MAPPO-night-{arm}-rho{load:g}-seed0",
                ),
            )
            (out / f"rl-rho{load:g}-{arm}-process.json").write_text(
                json.dumps({"pid": child.pid, "args": cmd})
            )
            if child.wait():
                raise RuntimeError(f"RL failed: {load}/{arm}; saved checkpoint retained")
        print("RL DONE", load, arm, flush=True)
        return load, arm, folder

    for load, cfg in configs.items():
        (out / f"scenario-rho{load:g}.json").write_text(cfg.model_dump_json(indent=2))
    jobs = [(load, arm) for load in LOADS for arm in ["mixed", "single", "scratch"]]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for load, arm, folder in pool.map(run_arm, jobs):
            frames = episodes * configs[load].cycles
            report = json.loads((folder / f"evaluation-stochastic-{frames}.json").read_text())
            assert report["seeds"] == seeds
            rows.append(dict(load=load, method=arm, folder=str(folder), **report["mean"]))
            (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    # Matched workloads for final comparisons, irrespective of completion order.
    for load in LOADS:
        reports = [
            json.loads(
                (
                    Path(r["folder"])
                    / f"evaluation-stochastic-{episodes * configs[load].cycles}.json"
                ).read_text()
            )
            for r in rows
            if r["load"] == load
        ]
        reference = [e["workload"] for e in reports[0]["episodes"]]
        assert all([e["workload"] for e in r["episodes"]] == reference for r in reports)
    with (out / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with wandb.init(
        project="ecsai-deppo",
        name="Legacy-mixed-load-final-v1",
        mode=args.wandb_mode,
        dir=str(out),
        config=spec,
    ) as run:
        keys = ["load", "method", "success_rate", "mean_latency_s", "episode_return"]
        run.log(
            {"comparison": wandb.Table(columns=keys, data=[[r[k] for k in keys] for r in rows])}
        )
    (out / "complete.json").write_text(json.dumps({"runs": len(rows), "validation_only": True}))


if __name__ == "__main__":
    main()
