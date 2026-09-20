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
        "--method", choices=["DEPPO-adapted", "MAPPO-no-context"], default="DEPPO-adapted"
    )
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--eval-interval", type=int, default=8192)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--resume", type=Path)
    p = sub.choices["evaluate"]
    p.add_argument("--method", choices=["random", "local", "forward"], default="local")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=10)
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
        )
    elif args.command == "evaluate":
        from .experiment import evaluate, save_report

        policy = None
        method = args.method
        if args.checkpoint:
            import torch

            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            config, policy, method = (
                ScenarioConfig(**payload["scenario"]),
                payload["policy"],
                payload["method"],
            )
        result = evaluate(
            config,
            method,
            policy,
            range(1_000_000_000 + args.seed, 1_000_000_000 + args.seed + args.episodes),
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
