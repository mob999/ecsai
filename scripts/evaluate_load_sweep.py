"""Frozen-checkpoint load sweep with fixed physical capacity and bounded retries."""

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from edge_sim_learning.experiment import evaluate, save_report
from edge_sim_learning.pretrain import sha256
from edge_sim_learning.scenario import ScenarioConfig, build_run

METHODS = [
    ("MAPPO", "MAPPO-no-context", None),
    ("DD", "DD-adapted", None),
    ("Random", "random", None),
    ("Always-forward", "forward", None),
    ("Always-local", "local", None),
    ("Forward-r0.6", "forward", 0.6),
    ("Queue-adaptive", "queue-adaptive", None),
]


def load_config(base, load):
    if base.capacity_profile != "calibrated" or base.bandwidth_mode != "shared":
        raise ValueError("sweep requires calibrated shared capacity")
    return ScenarioConfig.model_validate(
        base.model_dump()
        | {
            "request_rate": base.request_rate * load / base.delivery_load,
            "delivery_load": load,
            "max_retries": 2,
            "retry_delay_s": 0.1,
        }
    )


def capacity_signature(cfg):
    run, _ = build_run(cfg, 42)
    return [(c.node_id, c.cluster_id, c.total_bandwidth_bytes_s) for c in run.content.caches]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=["small", "medium", "large"], required=True)
    parser.add_argument("--loads", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.25])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for load in args.loads:
        reference = None
        for label, method, ratio in METHODS:
            source = args.root / f"scale-{args.size}" / label
            base = ScenarioConfig.model_validate_json((source / "scenario.json").read_text())
            cfg = load_config(base, load)
            before, after = capacity_signature(base), capacity_signature(cfg)
            assert [x[:2] for x in before] == [x[:2] for x in after]
            assert np.allclose([x[2] for x in before], [x[2] for x in after], rtol=1e-12)
            checkpoint, policy = None, None
            if label in {"MAPPO", "DD"}:
                checkpoint = source / "last.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                policy = payload["policy"]
                frames = payload["experiment"]["state"]["total_frames"]
                original = source / f"evaluation-stochastic-{frames}.json"
            else:
                original = source / "evaluation.json"
            seeds = json.loads(original.read_text())["seeds"]
            if args.smoke:
                seeds = seeds[:1]
            folder = args.output / f"load-{load:g}" / label
            folder.mkdir(parents=True, exist_ok=True)
            spec = dict(
                scenario=cfg.model_dump(),
                seeds=seeds,
                method=method,
                fixed_ratio=ratio,
                capacities=after,
                checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
            )
            spec = json.loads(json.dumps(spec))
            sp = folder / "spec.json"
            if sp.exists() and json.loads(sp.read_text()) != spec:
                raise ValueError(f"changed evaluation spec: {folder}")
            sp.write_text(json.dumps(spec, indent=2))
            dest = folder / "evaluation.json"
            print(f"START {args.size}/{load}/{label}", flush=True)
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
                # Keep complete attempt audit compressed; compact episode metrics remain readable.
                audit = folder / "attempts.jsonl.gz"
                with gzip.open(audit.with_suffix(".tmp"), "wt") as stream:
                    for ep in result["episodes"]:
                        stream.write(
                            json.dumps({"seed": ep["seed"], "requests": ep.pop("retry_outcomes")})
                            + "\n"
                        )
                audit.with_suffix(".tmp").replace(audit)
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(json.dumps(result))
                tmp.replace(dest)
                os.environ["WANDB_RUN_GROUP"] = f"load-sweep-{args.size}"
                os.environ["ECSAI_RUN_NAME"] = f"{label}-rho{load:g}"
                save_report(folder, cfg, 0, label, "offline", [result["mean"]])
            workloads = []
            for ep in result["episodes"]:
                assert ep["logical_unfinished"] == ep["unfinished"] == 0
                assert ep["logical_requests"] == ep["logical_completed"] + ep["logical_failed"]
                assert ep["arrived"] == ep["completed"] + ep["timed_out"] + ep["rejected"]
                workloads.append((ep["seed"], ep["workload"], ep["logical_requests"]))
            if reference is None:
                reference = workloads
            assert reference == workloads
            rows.append(
                dict(
                    size=args.size,
                    load=load,
                    method=label,
                    request_rate=cfg.request_rate,
                    **result["mean"],
                )
            )
            (args.output / "comparison.json").write_text(json.dumps(rows, indent=2))
            print(
                f"DONE {args.size}/{load}/{label}: "
                f"success={result['mean']['logical_success_rate']:.4f}",
                flush=True,
            )
    (args.output / "complete.json").write_text(json.dumps({"verified_cases": len(rows)}))


if __name__ == "__main__":
    main()
