"""Keep imports lazy: spawned SimGrid workers must not import the training stack."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DEPPO-adapted BenchMARL experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=["smoke", "small", "medium", "large"], default="small")
        p.add_argument("--scenario", type=Path, help="JSON ScenarioConfig overrides profile")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--wandb-mode", choices=["offline", "online"], required=True)
    p = sub.choices["train"]
    p.add_argument(
        "--method",
        choices=["DEPPO-adapted", "MAPPO-no-context", "DD-adapted"],
        default="DEPPO-adapted",
    )
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--normalize-advantage", action="store_true")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--initial-std", type=float)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--eval-interval", type=int, default=8192)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-workers", type=int, default=1)
    p.add_argument(
        "--eval-stochastic",
        action="store_true",
        help="Also log sampled-policy validation; best checkpoint still uses deterministic scores",
    )
    p.add_argument("--resume", type=Path)
    p = sub.choices["evaluate"]
    p.add_argument(
        "--method", choices=["random", "local", "forward", "queue-adaptive"], default="local"
    )
    p.add_argument(
        "--exploration", choices=["deterministic", "stochastic"], default="deterministic"
    )
    p.add_argument("--fixed-ratio", type=float)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=1, help="Parallel evaluation episodes")
    p = sub.choices["benchmark"]
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    from .scenario import ScenarioConfig

    config = (
        ScenarioConfig.model_validate_json(args.scenario.read_text())
        if args.scenario
        else ScenarioConfig.profile(args.profile)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        from .experiment import train

        train(
            config,
            args.output,
            args.method,
            args.seed,
            args.episodes,
            args.workers,
            args.batch,
            args.epochs,
            args.minibatch,
            args.device,
            args.wandb_mode,
            args.eval_interval,
            args.eval_episodes,
            args.resume,
            args.learning_rate,
            normalize_advantage=args.normalize_advantage,
            hidden_size=args.hidden_size,
            context_size=args.context_size,
            initial_std=args.initial_std,
            eval_stochastic=args.eval_stochastic,
            eval_workers=args.eval_workers,
        )
    elif args.command == "evaluate":
        from .experiment import evaluate, save_report

        policy = None
        method = args.method
        if args.checkpoint:
            import torch

            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            expected_dim = (
                payload["scenario"]["scheduler_capacity"] + 2
                if payload["method"] == "DD-adapted"
                else 5
            )
            if payload.get("action_dim") != expected_dim:
                raise ValueError("checkpoint action dimension mismatch")
            config, policy, method = (
                ScenarioConfig(**payload["scenario"]),
                payload["policy"],
                payload["method"],
            )
        base_seed = 1_000_000_000 if args.split == "validation" else 2_000_000_000
        result = evaluate(
            config,
            method,
            policy,
            range(base_seed + args.seed, base_seed + args.seed + args.episodes),
            fixed_ratio=args.fixed_ratio,
            exploration=args.exploration,
            workers=args.workers,
        )
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
        save_report(args.output, config, args.seed, method, args.wandb_mode, [result["mean"]])
        print(json.dumps(result["mean"], indent=2))
    else:
        from .benchmark import benchmark
        from .experiment import save_report

        result = benchmark(config, args.workers, args.steps, args.seed)
        (args.output / "benchmark.json").write_text(json.dumps(result, indent=2))
        save_report(args.output, config, args.seed, "benchmark", args.wandb_mode, result["results"])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
