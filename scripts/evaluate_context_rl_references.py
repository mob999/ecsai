"""Frozen BC and ordinary references on the contextual RL validation workloads."""

import argparse
import csv
import json
from pathlib import Path

from edge_sim_learning import multiscale_bc as bc
from edge_sim_learning.scenario import ScenarioConfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("episodes and workers must be positive")
    cfg = ScenarioConfig.model_validate_json(args.scenario.read_text()).model_copy(
        update={"episode_loads": (0.75,)}
    )
    seeds = list(range(1_100_000_000, 1_100_000_000 + args.episodes))
    bc.EVAL_WORKERS = args.workers
    rows = []
    for name, method, ratio, checkpoint in [
        ("Random", "random", None, None),
        ("Always-local", "local", 0.5, None),
        ("Always-forward", "forward", 0.5, None),
        ("Forward-r0.6", "forward", 0.6, None),
        ("Queue-adaptive", "queue-adaptive", None, None),
        ("BC-frozen", "MAPPO-no-context", None, args.base),
    ]:
        with bc.evaluation_slot():
            result = bc.cached_evaluate(
                args.output / name, cfg, seeds, method, ratio, checkpoint
            )
        rows.append(dict(method=name, **result["mean"]))
        print(name, result["mean"]["logical_success_rate"], flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
    with (args.output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
