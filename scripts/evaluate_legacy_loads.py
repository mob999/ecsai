"""Frozen legacy policies: fixed-capacity load sweep, without retries or training."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import wandb
from edge_sim_learning.experiment import evaluate
from edge_sim_learning.pretrain import sha256
from edge_sim_learning.scenario import ScenarioConfig, build_run


def load_config(base, load):
    return ScenarioConfig.model_validate(
        base.model_dump()
        | {"request_rate": base.request_rate * load / base.delivery_load, "delivery_load": load}
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--dd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--loads", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1, 1.25])
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="online")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    policies = []
    base = None
    sources = []
    for label, path, method in [
        ("MAPPO-pretrained", args.pretrained, "MAPPO-no-context"),
        ("MAPPO-scratch", args.scratch, "MAPPO-no-context"),
        ("DD-adapted", args.dd, "DD-adapted"),
    ]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ScenarioConfig.model_validate(payload["scenario"])
        if base is None:
            base = cfg
        if cfg != base or payload["method"] != method:
            raise ValueError("checkpoint scenarios or methods do not match")
        if cfg.max_retries or cfg.episode_loads or cfg.capacity_profile != "calibrated":
            raise ValueError("requires fixed-load calibrated legacy scenario without retries")
        policies.append((label, method, payload["policy"], None))
        sources.append(
            {
                "method": label,
                "path": str(path),
                "sha256": sha256(path),
                "frames": payload["experiment"]["state"]["total_frames"],
            }
        )
    policies += [
        (label, method, None, ratio)
        for label, method, ratio in [
            ("Random", "random", None),
            ("Always-local", "local", 0.5),
            ("Always-forward", "forward", 0.5),
            ("Forward-r0.6", "forward", 0.6),
            ("Queue-adaptive", "queue-adaptive", None),
        ]
    ]
    seeds = list(range(1_000_000_000, 1_000_000_000 + args.episodes))
    spec = {
        "sources": sources,
        "base": base.model_dump(),
        "loads": args.loads,
        "seeds": seeds,
        "split": "development validation",
        "exploration": "stochastic",
    }
    spec = json.loads(json.dumps(spec))
    sp = args.output / "spec.json"
    if sp.exists() and json.loads(sp.read_text()) != spec:
        raise ValueError("refusing to mix different sweep configurations")
    sp.write_text(json.dumps(spec, indent=2))
    run = wandb.init(
        project="ecsai-deppo",
        name="Legacy-frozen-load-sweep",
        mode=args.wandb_mode,
        dir=str(args.output),
        config=spec,
    )
    (args.output / "wandb.json").write_text(json.dumps({"url": run.url}))
    rows = []
    physical, _ = build_run(base, 42)
    capacities = [c.total_bandwidth_bytes_s for c in physical.content.caches]
    try:
        for load in args.loads:
            cfg = load_config(base, load)
            physical, _ = build_run(cfg, 42)
            assert np.allclose(
                capacities, [c.total_bandwidth_bytes_s for c in physical.content.caches], rtol=1e-12
            )
            reference = None
            for label, method, policy, ratio in policies:
                folder = args.output / f"rho-{load:g}" / label
                folder.mkdir(parents=True, exist_ok=True)
                dest = folder / "evaluation.json"
                print(f"START rho={load:g} {label}", flush=True)
                if dest.exists():
                    result = json.loads(dest.read_text())
                else:
                    result = evaluate(
                        cfg,
                        method,
                        policy,
                        seeds,
                        fixed_ratio=ratio,
                        exploration="stochastic",
                        workers=args.workers,
                    )
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_text(json.dumps(result))
                    tmp.replace(dest)
                workloads = [ep["workload"] for ep in result["episodes"]]
                assert result["seeds"] == seeds
                if reference is None:
                    reference = workloads
                assert workloads == reference
                assert all(ep["unfinished"] == 0 for ep in result["episodes"])
                row = {
                    "load": load,
                    "method": label,
                    "request_rate": cfg.request_rate,
                    **result["mean"],
                }
                rows.append(row)
                (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
                with (args.output / "summary.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                run.log({"load": load, **{f"{label}/{k}": v for k, v in result["mean"].items()}})
                print(
                    f"DONE rho={load:g} {label} success={row['success_rate']:.5f} "
                    f"latency={row['mean_latency_s']:.5f}",
                    flush=True,
                )
        plots = {}
        for key in [
            "success_rate",
            "mean_latency_s",
            "episode_return",
            "latency_p95_s",
            "timeout_rate",
            "rejection_rate",
        ]:
            names = [p[0] for p in policies]
            ys = [[r[key] for r in rows if r["method"] == name] for name in names]
            plots[f"comparison/{key}"] = wandb.plot.line_series(
                xs=args.loads, ys=ys, keys=names, title=key, xname="load"
            )
        run.log(plots)
        artifact = wandb.Artifact("legacy-frozen-load-results", type="evaluation")
        for name in ["summary.csv", "summary.json", "spec.json"]:
            artifact.add_file(str(args.output / name))
        run.log_artifact(artifact)
    finally:
        run.finish()
    (args.output / "complete.json").write_text(json.dumps({"cases": len(rows)}))


if __name__ == "__main__":
    main()
