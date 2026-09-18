"""Run: uv run python -m examples.external_preemption."""

from edge_sim import start
from edge_sim_models import Defer, Place, Resume, RunSpec, Suspend

from .scenarios import preemption


def main() -> None:
    run = RunSpec(scenario=preemption(), run_id="external-preemption", external=True)
    phase = 0
    with start(run) as session:
        for _ in range(20):
            step = session.advance()
            if step.kind == "finished":
                result = session.result()
                break
            if step.kind == "time":
                continue
            decision = step.decision
            assert decision is not None
            stages = {stage.request_id: stage for stage in decision.view.stages}
            if phase == 0:
                commands = (
                    Place(request_id="background", stage_id="compute", node_id="n0"),
                    Defer(until_s=1),
                )
            elif phase == 1:
                assert stages["background"].status == "RUNNING"
                commands = (
                    Suspend(request_id="background", stage_id="compute"),
                    Place(request_id="urgent", stage_id="compute", node_id="n0"),
                    Defer(until_s=2),
                )
            elif phase == 2:
                assert stages["urgent"].status == "SUCCEEDED"
                assert stages["background"].status == "SUSPENDED"
                # The completed urgent stage released memory; suspended work retains its 1 MB.
                assert decision.view.nodes[0].memory_used_bytes == 1_000_000
                commands = (
                    Resume(request_id="background", stage_id="compute"),
                    Defer(until_s=step.time_s + stages["background"].remaining_flops / 1e9),
                )
            else:
                raise RuntimeError("Unexpected decision after background work was resumed")
            session.apply(decision.decision_id, commands)
            phase += 1
        else:
            raise RuntimeError("External policy exceeded its decision guard")
        assert phase == 3
        assert result.completed
        requests = {request.request_id: request for request in result.state.requests}
        assert all(request.status == "SUCCEEDED" for request in requests.values())
        assert requests["urgent"].completed_s < requests["background"].completed_s
        assert len(session.commands()) == 3
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
