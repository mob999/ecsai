"""Train independent scale-specific MAPPO policies with or without BC initialization."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge_sim_learning.multiscale_bc import freeze_spec, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, base, output = args.source.resolve(), args.base.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((source / "spec.json").read_text())
    freeze_spec(
        output / "comparison-spec.json",
        dict(
            base_sha256=sha256(base),
            scales=["small", "medium", "large"],
            loads=[0.25, 0.5, 0.75, 1, 1.25],
            seed=0,
            env_steps=65536,
            workers_per_run=2,
            training_concurrency=6,
            evaluation_concurrency=1,
            reward="business",
            fixed_scale=0.1,
            actor="independent 4x256 LayerNorm, dropout disabled in both arms",
            critic="shared 2x256",
            actor_initialization="pretrained or scratch; critic initialization identical",
        ),
    )
    jobs = []
    for size in ["small", "medium", "large"]:
        cfg = spec["conditions"][size + "-rho0.75"] | {"reward_mode": "business", "cycles": 128}
        path = output / (size + ".json")
        freeze_spec(path, cfg)
        for arm in ["pretrained", "scratch"]:
            dest = output / (size + "-" + arm)
            cmd = [
                sys.executable,
                "-m",
                "edge_sim_learning.cli",
                "train",
                "--scenario",
                str(path),
                "--output",
                str(dest),
                "--method",
                "MAPPO-no-context",
                "--episodes",
                "512",
                "--workers",
                "2",
                "--batch",
                "512",
                "--minibatch",
                "512",
                "--epochs",
                "5",
                "--learning-rate",
                "0.0001",
                "--hidden-size",
                "256",
                "--local-context",
                "--load-mix",
                "0.25",
                "0.5",
                "0.75",
                "1",
                "1.25",
                "--device",
                "cuda",
                "--wandb-mode",
                "online",
                "--eval-interval",
                "8192",
                "--eval-episodes",
                "1",
                "--eval-workers",
                "2",
            ]
            if arm == "pretrained":
                cmd += ["--actor-init", str(base)]
            jobs.append((size + "-" + arm, cmd))

    def run(job):
        name, cmd = job
        if (output / name / "last.pt").exists():
            raise RuntimeError(f"{name} already started: inspect checkpoint before restarting")
        with (output / (name + ".log")).open("a") as log:
            child = subprocess.Popen(
                cmd,
                env=dict(
                    os.environ,
                    ECSAI_RUN_NAME="MAPPO-context-" + name + "-seed0",
                    BC_EVAL_LOCK=str(output / "evaluation.lock"),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            (output / (name + "-process.json")).write_text(
                json.dumps(dict(pid=child.pid, args=cmd))
            )
            code = child.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; checkpoint retained")
        return name

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in pool.map(run, jobs):
            print("COMPLETE", name, flush=True)


if __name__ == "__main__":
    main()
