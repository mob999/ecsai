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
    parser.add_argument("command", choices=["validate", "run", "visualize"])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commands", type=Path, help="Command log JSON for visualization")
    args = parser.parse_args()
    if args.command == "visualize":
        from .visualization import visualize_file

        try:
            target = visualize_file(
                args.scenario, args.output or args.scenario.with_suffix(".html"), args.commands
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(target.resolve())
        return
    if args.commands:
        parser.error("--commands is only supported by visualize")
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
