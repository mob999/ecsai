"""Execution helper for examples that use the built-in policy."""

from edge_sim import start
from edge_sim_models import PolicySpec, RunResult, RunSpec, ScenarioSpec


def run_example(scenario: ScenarioSpec, name: str) -> RunResult:
    with start(RunSpec(scenario=scenario, run_id=name, policy=PolicySpec(name="fifo"))) as session:
        while True:
            step = session.advance()
            if step.kind == "finished":
                result = session.result()
                break
            if step.kind == "decision":
                raise RuntimeError("Built-in policy unexpectedly requested an external decision")
        if not result.completed or any(r.status != "SUCCEEDED" for r in result.state.requests):
            raise RuntimeError(f"Example did not complete successfully: {result}")
        print(result.model_dump_json(indent=2))
        return result
