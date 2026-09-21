"""Re-evaluate frozen scale and transfer policies with real bounded request retries."""

import argparse
import json
import os
from pathlib import Path

import torch
from edge_sim_learning.experiment import evaluate, save_report
from edge_sim_learning.pretrain import load_base_policy, sha256
from edge_sim_learning.scenario import ScenarioConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--scale-root", type=Path)
    p.add_argument("--transfer-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-delay-s", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not args.scale_root and not args.transfer_root:
        raise ValueError("provide at least one original experiment root")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, references = [], {}

    def case(
        group, label, cfg, seeds, original, policy=None, method="local", ratio=None, checkpoint=None
    ):
        cfg = ScenarioConfig.model_validate(
            cfg.model_dump()
            | {
                "max_retries": args.max_retries,
                "retry_delay_s": args.retry_delay_s,
            }
        )
        folder = args.output / group / label
        folder.mkdir(parents=True, exist_ok=True)
        spec = dict(
            scenario=cfg.model_dump(),
            seeds=seeds,
            method=method,
            fixed_ratio=ratio,
            checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
            original_evaluation_sha256=sha256(original),
        )
        spec_path = folder / "spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError(f"changed evaluation specification: {folder}")
        spec_path.write_text(json.dumps(spec, indent=2))
        destination = folder / "evaluation.json"
        if destination.exists():
            result = json.loads(destination.read_text())
        else:
            print(f"START {group}/{label}", flush=True)
            result = evaluate(
                cfg,
                method,
                policy,
                seeds,
                fixed_ratio=ratio,
                exploration="stochastic",
                workers=args.workers,
            )
            temp = destination.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result))
            temp.replace(destination)
            os.environ["WANDB_RUN_GROUP"] = f"retry-{group}"
            os.environ["ECSAI_RUN_NAME"] = label
            save_report(folder, cfg, 0, label, "offline", [result["mean"]])
        previous = json.loads(original.read_text())
        expected = {e["seed"]: e for e in previous["episodes"]}
        workloads = []
        for episode in result["episodes"]:
            old = expected[episode["seed"]]
            assert episode["workload"] == old["workload"]
            assert episode["logical_requests"] == old["arrived"]
            assert episode["logical_unfinished"] == episode["unfinished"] == 0
            assert (
                episode["logical_requests"]
                == episode["logical_completed"] + episode["logical_failed"]
            )
            assert (
                episode["arrived"]
                == episode["completed"] + episode["timed_out"] + episode["rejected"]
            )
            workloads.append((episode["seed"], episode["workload"], episode["logical_requests"]))
        if group in references:
            assert workloads == references[group]
        else:
            references[group] = workloads
        rows.append({"group": group, "method": label, **result["mean"]})
        (args.output / "comparison.json").write_text(json.dumps(rows, indent=2))
        print(
            f"DONE {group}/{label}: success={result['mean']['logical_success_rate']:.4f}, "
            f"resolution={result['mean']['mean_resolution_time_s']:.4f}s",
            flush=True,
        )

    if args.scale_root:
        for size in ("small", "medium", "large"):
            for label, method, ratio in [
                ("MAPPO", "MAPPO-no-context", None),
                ("DD", "DD-adapted", None),
                ("Random", "random", None),
                ("Always-forward", "forward", None),
                ("Always-local", "local", None),
                ("Forward-r0.6", "forward", 0.6),
                ("Queue-adaptive", "queue-adaptive", None),
            ]:
                root = args.scale_root / f"scale-{size}" / label
                cfg = ScenarioConfig.model_validate_json((root / "scenario.json").read_text())
                policy, checkpoint = None, None
                if label in {"MAPPO", "DD"}:
                    checkpoint = root / "last.pt"
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    policy = payload["policy"]
                    frames = payload["experiment"]["state"]["total_frames"]
                    original = root / f"evaluation-stochastic-{frames}.json"
                else:
                    original = root / "evaluation.json"
                seeds = json.loads(original.read_text())["seeds"]
                case(
                    f"scale-{size}",
                    label,
                    cfg,
                    seeds[:1] if args.smoke else seeds,
                    original,
                    policy,
                    method,
                    ratio,
                    checkpoint,
                )
    if args.transfer_root:
        root = args.transfer_root
        cfg = ScenarioConfig.model_validate_json((root / "target.json").read_text())
        for label in ("BC-only", "Scratch", "BC-full", "BC-head"):
            if label == "BC-only":
                checkpoint = root / "base/best.pt"
                policy = load_base_policy(checkpoint)
                original = root / label / "evaluation.json"
            else:
                checkpoint = root / label / "best-stochastic.pt"
                policy = torch.load(checkpoint, map_location="cpu", weights_only=False)["policy"]
                original = root / f"{label}-test/evaluation.json"
            seeds = json.loads(original.read_text())["seeds"]
            case(
                "transfer",
                label,
                cfg,
                seeds[:1] if args.smoke else seeds,
                original,
                policy,
                "MAPPO-no-context",
                checkpoint=checkpoint,
            )
    (args.output / "complete.json").write_text(json.dumps({"verified_cases": len(rows)}))


if __name__ == "__main__":
    main()
