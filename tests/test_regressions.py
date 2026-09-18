"""Correctness regressions exercised through spawned, real SimGrid workers."""

import pytest
from edge_sim import start
from edge_sim_models import (
    ArtifactSpec,
    InputBinding,
    LinkSpec,
    NodeSpec,
    PolicySpec,
    RequestSpec,
    RouteSpec,
    RunSpec,
    ScenarioSpec,
    StageSpec,
    WorkflowArtifactSpec,
    WorkflowSpec,
)

pytestmark = pytest.mark.integration


def node(name, storage=100):
    return NodeSpec(id=name, speed_flops=10, memory_bytes=100, storage_bytes=storage)


def zero_delivery(latencies, deadlines=(None,), produced=False):
    links = tuple(
        LinkSpec(id=f"l{i}", bandwidth_bytes_s=100, latency_s=latency)
        for i, latency in enumerate(latencies)
    )
    return ScenarioSpec(
        nodes=(node("a"), node("b")),
        links=links,
        routes=(RouteSpec(src="a", dst="b", links=tuple(link.id for link in links)),),
        artifacts=() if produced else (ArtifactSpec(id="global", size_bytes=0, locations=("a",)),),
        workflows=(
            WorkflowSpec(
                id="w",
                stages=(StageSpec(id="p", flops=20, outputs=("x",), eligible_nodes=("a",)),)
                if produced
                else (),
                artifacts=(
                    WorkflowArtifactSpec(
                        id="x",
                        size_bytes=0,
                        producer_stage="p" if produced else None,
                    ),
                ),
            ),
        ),
        requests=tuple(
            RequestSpec(
                id=f"r{i}",
                workflow="w",
                receiver="b",
                deadline_s=deadline,
                input_bindings=()
                if produced
                else (InputBinding(artifact_id="x", global_artifact_id="global"),),
            )
            for i, deadline in enumerate(deadlines)
        ),
    )


@pytest.mark.parametrize("latencies", [(0,), (0.25, 0.75)])
@pytest.mark.parametrize("produced", [False, True])
def test_zero_byte_delivery_preserves_latency_and_exact_deadline(latencies, produced):
    expected = sum(latencies) + (2 if produced else 0)
    scenario = zero_delivery(latencies, (expected,), produced)
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        result = session.advance().result
        assert result.metrics.completed == 1
        assert result.now_s == pytest.approx(expected)
        assert result.metrics.transfers == 1
        assert all(m.value == 0 for m in result.metrics.link_utilization)
        transfers = [e for e in result.events if e.kind.startswith("transfer_")]
        assert [e.kind for e in transfers] == ["transfer_started", "transfer_finished"]
        assert transfers[1].time_s - transfers[0].time_s == pytest.approx(sum(latencies))


def test_zero_byte_transfer_coalesces_and_survives_one_waiter_timeout():
    scenario = zero_delivery((0.25, 0.75), (0.5, 1))
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        boundary = session.advance(until_time=0.5)
        assert boundary.kind == "time"
        assert [r.status for r in boundary.view.requests] == ["TIMED_OUT", "ACTIVE"]
        assert all("b" not in a.locations for a in boundary.view.artifacts)
        result = session.advance().result
        assert result.now_s == 1
        assert result.metrics.transfers == 1
        assert result.metrics.completed == result.metrics.timed_out == 1


def test_zero_byte_transfer_cancels_when_last_waiter_times_out():
    scenario = zero_delivery((2,), (0.5,))
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        result = session.advance().result
        assert result.now_s == 0.5
        assert result.metrics.timed_out == 1
        assert all(e.kind != "transfer_finished" for e in result.events)
        assert all(n.storage_used_bytes == 0 for n in result.state.nodes)


def test_rejection_discards_earlier_placements_and_other_requests_continue():
    scenario = ScenarioSpec(
        nodes=(node("a"),),
        workflows=(
            WorkflowSpec(
                id="bad",
                stages=(
                    StageSpec(id="a", flops=10),
                    StageSpec(id="z", flops=10, memory_bytes=101),
                ),
            ),
            WorkflowSpec(id="good", stages=(StageSpec(id="work", flops=10),)),
        ),
        requests=(
            RequestSpec(id="bad", workflow="bad", receiver="a"),
            RequestSpec(id="good", workflow="good", receiver="a"),
        ),
    )
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        result = session.advance().result
        assert result.metrics.rejected == result.metrics.completed == 1
        assert result.now_s == 1
        assert all(c.request_id == "good" for record in session.commands() for c in record.commands)
        assert session.advance().kind == "finished"


def pipeline(cacheable=True, directed_consumer=False):
    nodes = (node("a", 120 if directed_consumer else 100), node("b", 200))
    routes = (RouteSpec(src="a", dst="b", links=("wire",)),)
    stages = (
        StageSpec(id="p", flops=10, outputs=("x",), eligible_nodes=("a",)),
        StageSpec(id="q", flops=10, inputs=("x",), eligible_nodes=("b",)),
        StageSpec(id="r", flops=10, depends_on=("q",), outputs=("y",), eligible_nodes=("a",)),
        StageSpec(id="s", flops=10, inputs=("x", "y"), eligible_nodes=("b",)),
    )
    if directed_consumer:
        nodes += (node("c"),)
        routes += (RouteSpec(src="a", dst="c", links=("wire",)),)
        stages += (
            StageSpec(
                id="t",
                flops=10,
                inputs=("x",),
                depends_on=("s",),
                eligible_nodes=("c",),
            ),
        )
    return ScenarioSpec(
        nodes=nodes,
        links=(LinkSpec(id="wire", bandwidth_bytes_s=60),),
        routes=routes,
        workflows=(
            WorkflowSpec(
                id="w",
                stages=stages,
                artifacts=(
                    WorkflowArtifactSpec(
                        id="x", size_bytes=60, producer_stage="p", cacheable=cacheable
                    ),
                    WorkflowArtifactSpec(
                        id="y", size_bytes=60, producer_stage="r", cacheable=cacheable
                    ),
                ),
            ),
        ),
        requests=(RequestSpec(id="req", workflow="w", receiver="b"),),
    )


@pytest.mark.parametrize("cacheable", [False, True])
def test_redundant_replica_can_be_reclaimed_under_storage_pressure(cacheable):
    scenario = pipeline(cacheable)
    capacities = {n.id: n.storage_bytes for n in scenario.nodes}
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        for boundary in range(8):
            step = session.advance(until_time=boundary)
            assert all(n.storage_used_bytes <= capacities[n.node_id] for n in step.view.nodes)
            if step.kind == "finished":
                break
        assert step.kind == "finished"
        assert step.result.metrics.completed == 1
        assert step.time_s == 6
        assert step.result.metrics.transfers == 2


def test_noncacheable_replica_preserves_only_route_to_unplaced_consumer():
    with start(RunSpec(scenario=pipeline(False, True)), timeout_s=5) as session:
        step = session.advance(until_time=3)
        x = next(a for a in step.view.artifacts if a.artifact_id == "req/x")
        assert x.locations == ("a", "b")
        result = session.advance().result
        assert result.metrics.completed == 1
        assert result.now_s == 8
        assert result.metrics.transfers == 3


def test_edf_zero_deadline_precedes_no_deadline():
    scenario = ScenarioSpec(
        nodes=(node("a"),),
        workflows=tuple(
            WorkflowSpec(id=name, stages=(StageSpec(id="work", flops=flops),))
            for name, flops in (("instant", 0), ("long", 10))
        ),
        requests=(
            RequestSpec(id="a_long", workflow="long", receiver="a"),
            RequestSpec(id="z_due", workflow="instant", receiver="a", deadline_s=0),
        ),
    )
    with start(RunSpec(scenario=scenario, policy=PolicySpec(name="edf"))) as session:
        result = session.advance().result
        assert result.metrics.completed == 2
        assert result.state.requests[1].completed_s == 0


@pytest.mark.parametrize("failure", ["storage", "route"])
def test_impossible_content_receiver_is_rejected_at_arrival(failure):
    scenario = ScenarioSpec(
        nodes=(node("a", 200), node("b", 100)),
        links=(LinkSpec(id="wire", bandwidth_bytes_s=100),),
        routes=(RouteSpec(src="a", dst="b", links=("wire",)),) if failure == "storage" else (),
        artifacts=tuple(
            ArtifactSpec(id=name, size_bytes=60, locations=("a",)) for name in ("x", "y")
        ),
        workflows=(
            WorkflowSpec(
                id="w",
                stages=(),
                artifacts=tuple(
                    WorkflowArtifactSpec(id=name, size_bytes=60) for name in ("x", "y")
                ),
            ),
        ),
        requests=(
            RequestSpec(
                id="r",
                workflow="w",
                receiver="b",
                arrival_s=2,
                input_bindings=tuple(
                    InputBinding(artifact_id=name, global_artifact_id=name) for name in ("x", "y")
                ),
            ),
        ),
    )
    if failure == "route":
        scenario = scenario.model_copy(update={"nodes": (node("a", 200), node("b", 200))})
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        result = session.advance().result
        assert result.now_s == 2
        assert result.metrics.rejected == 1
        assert result.metrics.failed == result.metrics.transfers == 0
        assert result.state.requests[0].reason == "no_feasible_receiver"


def test_completion_then_deadline_then_arrival_at_same_time():
    scenario = ScenarioSpec(
        nodes=(node("a"),),
        workflows=tuple(
            WorkflowSpec(id=name, stages=(StageSpec(id="work", flops=work),))
            for name, work in (("short", 10), ("long", 20))
        ),
        requests=(
            RequestSpec(id="a_complete", workflow="short", receiver="a", deadline_s=1),
            RequestSpec(id="b_expire", workflow="long", receiver="a", deadline_s=1),
            RequestSpec(id="c_arrive", workflow="short", receiver="a", arrival_s=1),
        ),
    )
    with start(RunSpec(scenario=scenario), timeout_s=5) as session:
        result = session.advance().result
        boundary = [
            (e.kind, e.entity_id)
            for e in result.events
            if e.time_s == 1 and e.kind in {"request_finished", "request_arrived"}
        ]
        assert boundary == [
            ("request_finished", "a_complete"),
            ("request_finished", "b_expire"),
            ("request_arrived", "c_arrive"),
        ]
        assert result.metrics.completed == 2
        assert result.metrics.timed_out == 1
        assert result.now_s == 2


def test_edf_same_time_arrival_competes_before_dispatch():
    scenario = ScenarioSpec(
        nodes=(node("a"),),
        workflows=(WorkflowSpec(id="w", stages=(StageSpec(id="work", flops=10),)),),
        requests=(
            RequestSpec(id="a_running", workflow="w", receiver="a"),
            RequestSpec(id="b_queued", workflow="w", receiver="a"),
            RequestSpec(id="z_urgent", workflow="w", receiver="a", arrival_s=1, deadline_s=2),
        ),
    )
    with start(RunSpec(scenario=scenario, policy=PolicySpec(name="edf"))) as session:
        result = session.advance().result
        assert result.metrics.completed == 3
        assert result.state.stages[2].started_s == 1
        assert result.state.stages[1].started_s == 2
