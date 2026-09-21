"""Matched normalized-advantage learning-rate probes before full contextual RL."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(
        learning_rates=[1e-5, 3e-6], normalize_advantage=True, env_steps=8192,
        eval_episodes=4, eval_interval=4096, fixed_scale=0.1,
        reward="logical", scale="3/10", load=0.75, seed=0,
        purpose="Stability screening; not evidence of convergence by itself",
    )
    if (out / "probe-spec.json").exists():
        raise RuntimeError("Probe already launched; inspect checkpoints before resuming")
    (out / "probe-spec.json").write_text(json.dumps(spec, indent=2))

    def run(job):
        lr, arm = job
        name = f"lr{lr:g}-{arm}"
        cmd = [
            sys.executable, "-m", "edge_sim_learning.cli", "train",
            "--scenario", str(args.scenario.resolve()), "--output", str(out / name),
            "--method", "MAPPO-no-context", "--episodes", "64", "--workers", "2",
            "--batch", "512", "--minibatch", "512", "--epochs", "5",
            "--learning-rate", str(lr), "--normalize-advantage", "--hidden-size", "256",
            "--local-context", "--load-mix", "0.75", "--device", "cuda",
            "--wandb-mode", "online", "--eval-interval", "4096", "--eval-episodes", "4",
            "--eval-workers", "2",
        ]
        if arm == "pretrained":
            cmd += ["--actor-init", str(args.base.resolve())]
        with (out / f"{name}.log").open("a") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                BC_EVAL_LOCK=str(out / "evaluation.lock"),
                ECSAI_RUN_NAME=f"MAPPO-stability-{name}-seed0",
            ))
            (out / f"{name}-process.json").write_text(json.dumps(dict(pid=process.pid, args=cmd)))
            code = process.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; retain last checkpoint")
        return name

    jobs = [(lr, arm) for lr in spec["learning_rates"] for arm in ["pretrained", "scratch"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(run, jobs):
            print("COMPLETE", name, flush=True)


if __name__ == "__main__":
    main()
