"""Small analytical cases against the actual SimGrid engine, never a mock."""

import time

import pytest
from edge_sim import SDKError, start
from edge_sim.export import export_run
from edge_sim_models import (
    ArtifactSpec,
    Defer,
    InputBinding,
    LinkSpec,
    NodeSpec,
    Place,
    RequestSpec,
    Resume,
    RouteSpec,
    RunResult,
    RunSpec,
    ScenarioSpec,
    StageSpec,
    Suspend,
    WorkflowArtifactSpec,
    WorkflowSpec,
)

pytestmark = pytest.mark.integration


def node(name="edge", **kwargs):
    return NodeSpec(id=name, speed_flops=10, memory_bytes=100, storage_bytes=1000, **kwargs)


def compute_scenario(*, deadline=None, cores=1, memory=60):
    return ScenarioSpec(
        nodes=(node(cores=cores),),
        workflows=(
            WorkflowSpec(id="job", stages=(StageSpec(id="work", flops=20, memory_bytes=memory),)),
        ),
        requests=(RequestSpec(id="r", workflow="job", receiver="edge", deadline_s=deadline),),
    )


def finish(scenario, **kwargs):
    with start(RunSpec(scenario=scenario, **kwargs)) as session:
        step = session.advance()
        assert step.kind == "finished"
        return session.result()


def test_compute_and_exact_deadline():
    result = finish(compute_scenario(deadline=2))
    assert result.now_s == pytest.approx(2)
    assert result.metrics.completed == 1
    assert result.metrics.cpu_utilization[0].value == pytest.approx(1)
    assert result.state.nodes[0].memory_used_bytes == 0


def test_timeout_cleans_running_work_and_truncation_is_distinct():
    result = finish(compute_scenario(deadline=1))
    assert result.metrics.timed_out == 1
    assert result.state.stages[0].status == "CANCELLED"
    assert result.state.nodes[0].memory_used_bytes == 0
    truncated = finish(compute_scenario(), until_s=1)
    assert truncated.end_reason == "truncated"
    assert not truncated.completed
    assert truncated.metrics.unfinished == 1


def test_two_cores_still_obey_memory_admission():
    scenario = compute_scenario(cores=2)
    scenario = scenario.model_copy(
        update={
            "requests": (
                *scenario.requests,
                RequestSpec(id="s", workflow="job", receiver="edge"),
            )
        }
    )
    result = finish(scenario)
    assert result.now_s == pytest.approx(4)
    assert result.metrics.completed == 2
    assert result.metrics.cpu_utilization[0].value == pytest.approx(0.5)


def test_external_pause_preserves_progress_and_invalid_batch_is_atomic():
    with start(RunSpec(scenario=compute_scenario(), external=True)) as session:
        initial = session.advance()
        before = session.inspect()
        with pytest.raises(SDKError):
            session.apply(
                initial.decision.decision_id,
                (
                    Place(request_id="r", stage_id="work", node_id="edge"),
                    Suspend(request_id="r", stage_id="missing"),
                ),
            )
        assert session.inspect() == before
        session.apply(
            initial.decision.decision_id,
            (
                Place(request_id="r", stage_id="work", node_id="edge"),
                Defer(until_s=1),
            ),
        )
        running = session.advance()
        assert running.time_s == pytest.approx(1)
        assert running.view.stages[0].remaining_flops == pytest.approx(10)
        time.sleep(0.02)
        assert session.inspect().now_s == 1
        session.apply(
            running.decision.decision_id,
            (
                Suspend(request_id="r", stage_id="work"),
                Defer(until_s=3),
            ),
        )
        paused = session.advance()
        assert paused.view.stages[0].remaining_flops == pytest.approx(10)
        assert paused.view.nodes[0].memory_used_bytes == 60
        assert paused.view.nodes[0].cores_used == 0
        session.apply(paused.decision.decision_id, (Resume(request_id="r", stage_id="work"),))
        with pytest.raises(SDKError):
            session.apply(paused.decision.decision_id, (Resume(request_id="r", stage_id="work"),))
        step = session.advance()
        while step.kind != "finished":
            session.apply(step.decision.decision_id, (Defer(until_s=step.time_s + 10),))
            step = session.advance()
        assert step.time_s == pytest.approx(4)


def delivery_scenario(*, shared=True, arrival=0, size=100, cacheable=True):
    return ScenarioSpec(
        nodes=(node("cloud"), node()),
        links=(LinkSpec(id="wire", bandwidth_bytes_s=100),),
        routes=(RouteSpec(src="cloud", dst="edge", links=("wire",)),),
        artifacts=(
            ArtifactSpec(id="a", size_bytes=size, locations=("cloud",), cacheable=cacheable),
            ArtifactSpec(id="b", size_bytes=size, locations=("cloud",), cacheable=cacheable),
        ),
        workflows=(
            WorkflowSpec(
                id="fetch",
                stages=(),
                artifacts=(WorkflowArtifactSpec(id="input", size_bytes=size),),
            ),
        ),
        requests=(
            RequestSpec(
                id="r",
                workflow="fetch",
                receiver="edge",
                input_bindings=(InputBinding(artifact_id="input", global_artifact_id="a"),),
            ),
            RequestSpec(
                id="s",
                workflow="fetch",
                receiver="edge",
                arrival_s=arrival,
                input_bindings=(
                    InputBinding(artifact_id="input", global_artifact_id="a" if shared else "b"),
                ),
            ),
        ),
    )


def test_network_contention_and_inflight_coalescing():
    distinct = finish(delivery_scenario(shared=False))
    assert distinct.now_s == pytest.approx(2)
    assert distinct.metrics.transfers == 2
    assert distinct.metrics.link_utilization[0].value == pytest.approx(1)
    shared = finish(delivery_scenario())
    assert shared.now_s == pytest.approx(1)
    assert shared.metrics.transfers == 1
    assert shared.metrics.completed == 2


def test_later_cache_hit_and_noncacheable_cleanup():
    cached = finish(delivery_scenario(arrival=2))
    assert cached.metrics.transfers == 1
    uncached = finish(delivery_scenario(arrival=2, cacheable=False))
    assert uncached.metrics.transfers == 2
    assert uncached.state.nodes[1].storage_used_bytes == 0


def test_shared_transfer_survives_one_waiter_timeout():
    scenario = delivery_scenario()
    scenario = scenario.model_copy(
        update={
            "requests": (
                scenario.requests[0].model_copy(update={"deadline_s": 0.5}),
                scenario.requests[1],
            )
        }
    )
    result = finish(scenario)
    assert result.metrics.completed == 1
    assert result.metrics.timed_out == 1
    assert result.metrics.transfers == 1


def test_dag_data_moves_before_consumer_starts():
    scenario = ScenarioSpec(
        nodes=(node("a"), node("b")),
        links=(LinkSpec(id="wire", bandwidth_bytes_s=100),),
        routes=(RouteSpec(src="a", dst="b", links=("wire",)),),
        workflows=(
            WorkflowSpec(
                id="dag",
                artifacts=(
                    WorkflowArtifactSpec(id="middle", size_bytes=100, producer_stage="first"),
                ),
                stages=(
                    StageSpec(id="first", flops=10, outputs=("middle",), eligible_nodes=("a",)),
                    StageSpec(id="second", flops=10, inputs=("middle",), eligible_nodes=("b",)),
                ),
            ),
        ),
        requests=(RequestSpec(id="r", workflow="dag", receiver="b"),),
    )
    result = finish(scenario)
    assert result.now_s == pytest.approx(3)
    assert result.state.stages[1].started_s == pytest.approx(2)


def test_export_roundtrip_and_trace_disabled(tmp_path):
    import pyarrow.parquet as pq

    result = finish(compute_scenario())
    export_run(result, tmp_path)
    restored = RunResult.model_validate_json((tmp_path / "result.json").read_text())
    assert restored == result
    events = pq.read_table(tmp_path / "events.parquet")
    assert events.num_rows == len(result.events)
    assert events.schema.metadata[b"edge_sim.schema_version"] == b"1"
    quiet = finish(compute_scenario(), trace=False)
    assert not quiet.events
    assert quiet.metrics.completed == 1
