"""Run validated JSON scenarios using the standalone SDK."""

import argparse
import json
from pathlib import Path

from edge_sim_models import RunSpec, ScenarioSpec

from .export import export_run
from .session import start
from .settings import Settings


def main():
    parser = argparse.ArgumentParser(prog="edge-sim")
    parser.add_argument("command", choices=["validate", "run"])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    scenario = ScenarioSpec.model_validate_json(args.scenario.read_text())
    if args.command == "validate":
        print("Scenario valid")
        return
    settings = Settings()
    with start(RunSpec(scenario=scenario, seed=args.seed), timeout_s=settings.timeout_s) as session:
        step = session.advance()
        if step.kind != "finished":
            raise RuntimeError("CLI runs require an internal policy")
        result = session.result()
        export_run(result, args.output or settings.output_dir, session.commands())
        print(json.dumps(result.metrics.model_dump(mode="json"), indent=2))
