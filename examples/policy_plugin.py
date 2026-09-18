"""Run: uv run python -m examples.policy_plugin."""

from edge_sim import start
from edge_sim_models import DecisionCommand, DecisionRequest, Place, PluginSpec, PolicySpec, RunSpec
from edge_sim_models.policies import PlacementPolicy
from pydantic import BaseModel, ConfigDict

from .scenarios import content_delivery


class FirstCandidatePlacement(BaseModel):
    """An immutable Pydantic policy implementation with no engine references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def decide(self, decision: DecisionRequest) -> tuple[DecisionCommand, ...]:
        for candidate in decision.candidates:
            if candidate.nodes:
                return (
                    Place(
                        request_id=candidate.request_id,
                        stage_id=candidate.stage_id,
                        node_id=candidate.nodes[0],
                    ),
                )
        raise ValueError("Placement decision has no eligible candidate")


def make_policy() -> PlacementPolicy:
    return FirstCandidatePlacement()


def main() -> None:
    run = RunSpec(
        scenario=content_delivery(),
        run_id="plugin-delivery",
        policy=PolicySpec(
            name="fifo",
            plugins=(
                PluginSpec(
                    role="placement",
                    factory="examples.policy_plugin:make_policy",
                ),
            ),
        ),
    )
    with start(run) as session:
        step = session.advance()
        assert step.kind == "finished"
        result = session.result()
        assert result.completed
        assert all(request.status == "SUCCEEDED" for request in result.state.requests)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
