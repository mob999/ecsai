"""Domain invariants and the serialization boundary, without a simulator backend."""

import json

import pytest
from edge_sim_models import (
    AdvanceResult,
    ArtifactSpec,
    ArtifactState,
    Candidate,
    Decision,
    DecisionRequest,
    Defer,
    Detail,
    DomainEvent,
    Event,
    LinkSpec,
    Metric,
    NodeSpec,
    Place,
    Reject,
    RequestState,
    Resume,
    RunManifest,
    RunMetrics,
    RunResult,
    RunSpec,
    ScenarioSpec,
    StageSpec,
    StageState,
    StateView,
    Suspend,
    WorkflowSpec,
    validate,
)
from pydantic import TypeAdapter, ValidationError


@pytest.fixture
def scenario_data():
    return {
        "nodes": [
            {
                "id": "client",
                "role": "client",
                "speed_flops": 1,
                "memory_bytes": 0,
                "storage_bytes": 100,
            },
            {
                "id": "edge",
                "speed_flops": 10,
                "cores": 2,
                "memory_bytes": 100,
                "storage_bytes": 100,
            },
        ],
        "links": [{"id": "uplink", "bandwidth_bytes_s": 10, "latency_s": 0.1}],
        "routes": [{"src": "client", "dst": "edge", "links": ["uplink"]}],
        "artifacts": [{"id": "global", "size_bytes": 10, "locations": ["client"]}],
        "workflows": [
            {
                "id": "workflow",
                "artifacts": [
                    {"id": "input", "size_bytes": 10},
                    {"id": "middle", "size_bytes": 4, "producer_stage": "first"},
                    {"id": "output", "size_bytes": 1, "producer_stage": "last"},
                ],
                "stages": [
                    {
                        "id": "first",
                        "flops": 20,
                        "inputs": ["input"],
                        "outputs": ["middle"],
                        "eligible_nodes": ["edge"],
                    },
                    {
                        "id": "last",
                        "flops": 5,
                        "inputs": ["middle"],
                        "outputs": ["output"],
                        "depends_on": ["first"],
                    },
                ],
            }
        ],
        "requests": [
            {
                "id": "request",
                "workflow": "workflow",
                "receiver": "client",
                "arrival_s": 2,
                "deadline_s": 10,
                "input_bindings": [{"artifact_id": "input", "global_artifact_id": "global"}],
            }
        ],
    }


def test_scenario_roundtrip_is_deeply_immutable(scenario_data):
    scenario = validate(scenario_data)
    scenario_data["workflows"][0]["stages"][0]["inputs"].append("mutated")
    assert scenario.workflows[0].stages[0].inputs == ("input",)
    assert isinstance(scenario.requests[0].input_bindings, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        scenario.requests[0].input_bindings[0].artifact_id = "mutated"
    assert ScenarioSpec.model_validate_json(scenario.model_dump_json()) == scenario
    assert hash(scenario) == hash(validate(scenario))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_and_negative_physical_units_rejected(value):
    for model, data in [
        (StageSpec, {"id": "s", "flops": value}),
        (LinkSpec, {"id": "l", "bandwidth_bytes_s": value}),
        (LinkSpec, {"id": "l", "bandwidth_bytes_s": 1, "latency_s": value}),
        (Defer, {"until_s": value}),
        (Event, {"time_s": value, "kind": "arrival"}),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate(data)


@pytest.mark.parametrize("value", [-1, 1.5, True, "12", float("inf")])
def test_byte_counts_are_nonnegative_integers(value):
    with pytest.raises(ValidationError):
        ArtifactSpec(id="artifact", size_bytes=value)


@pytest.mark.parametrize("identifier", ["", "has space", "\t", "x\ny", "r/s", "/"])
def test_identifiers_cannot_be_empty_or_contain_whitespace(identifier):
    with pytest.raises(ValidationError):
        StageSpec(id=identifier, flops=0)


@pytest.mark.parametrize("collection", ["nodes", "links", "artifacts", "workflows", "requests"])
def test_scenario_ids_are_unique_in_each_namespace(scenario_data, collection):
    scenario_data[collection].append(scenario_data[collection][0])
    with pytest.raises(ValidationError, match="duplicate"):
        validate(scenario_data)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("routes", "src", "missing"),
        ("routes", "dst", "missing"),
        ("routes", "links", ["missing"]),
        ("artifacts", "locations", ["missing"]),
        ("requests", "workflow", "missing"),
        ("requests", "receiver", "missing"),
    ],
)
def test_scenario_references_must_resolve(scenario_data, target, field, value):
    scenario_data[target][0][field] = value
    with pytest.raises(ValidationError, match="unknown"):
        validate(scenario_data)


def test_directed_routes_cannot_be_ambiguous_or_empty(scenario_data):
    scenario_data["routes"].append(scenario_data["routes"][0])
    with pytest.raises(ValidationError, match="duplicate route"):
        validate(scenario_data)
    scenario_data["routes"].pop()
    scenario_data["routes"][0]["links"] = []
    with pytest.raises(ValidationError, match="requires links"):
        validate(scenario_data)
    scenario_data["routes"][0]["dst"] = "client"
    validate(scenario_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("depends_on", ["missing"]),
        ("inputs", ["missing"]),
        ("outputs", ["missing"]),
        ("eligible_nodes", ["missing"]),
    ],
)
def test_stage_references_must_resolve(scenario_data, field, value):
    scenario_data["workflows"][0]["stages"][0][field] = value
    with pytest.raises(ValidationError, match="unknown"):
        validate(scenario_data)


def test_explicit_and_artifact_dependency_cycles_rejected(scenario_data):
    workflow = scenario_data["workflows"][0]
    workflow["stages"][0]["depends_on"] = ["last"]
    with pytest.raises(ValidationError, match="cycle"):
        validate(scenario_data)
    workflow["stages"][0]["depends_on"] = []
    workflow["stages"][1]["depends_on"] = []
    workflow["stages"][0]["inputs"] = ["output"]
    with pytest.raises(ValidationError, match="cycle"):
        validate(scenario_data)


def test_fork_join_and_disconnected_stages_are_valid():
    workflow = WorkflowSpec(
        id="dag",
        stages=(
            StageSpec(id="root", flops=0),
            StageSpec(id="left", flops=1, depends_on=("root",)),
            StageSpec(id="right", flops=1, depends_on=("root",)),
            StageSpec(id="join", flops=1, depends_on=("left", "right")),
            StageSpec(id="independent", flops=0),
        ),
    )
    assert len(workflow.stages) == 5


def test_producers_must_match_outputs_and_be_unique(scenario_data):
    workflow = scenario_data["workflows"][0]
    workflow["artifacts"][1]["producer_stage"] = "last"
    with pytest.raises(ValidationError, match="producer must declare"):
        validate(scenario_data)
    workflow["artifacts"][1]["producer_stage"] = "first"
    workflow["stages"].append({"id": "other", "flops": 1, "outputs": ["middle"]})
    with pytest.raises(ValidationError, match="multiple producers"):
        validate(scenario_data)


@pytest.mark.parametrize(
    "bindings",
    [
        [],
        [{"artifact_id": "input", "global_artifact_id": "missing"}],
        [{"artifact_id": "middle", "global_artifact_id": "global"}],
        [{"artifact_id": "missing", "global_artifact_id": "global"}],
        [{"artifact_id": "input", "global_artifact_id": "global"}] * 2,
    ],
)
def test_bindings_exactly_cover_external_inputs(scenario_data, bindings):
    scenario_data["requests"][0]["input_bindings"] = bindings
    with pytest.raises(ValidationError):
        validate(scenario_data)


def test_bound_artifacts_need_matching_size_and_a_location(scenario_data):
    scenario_data["artifacts"][0]["size_bytes"] = 11
    with pytest.raises(ValidationError, match="sizes must match"):
        validate(scenario_data)
    scenario_data["artifacts"][0]["size_bytes"] = 10
    scenario_data["artifacts"][0]["locations"] = []
    with pytest.raises(ValidationError, match="initial location"):
        validate(scenario_data)


def test_request_deadline_cannot_precede_arrival(scenario_data):
    scenario_data["requests"][0]["deadline_s"] = 1
    with pytest.raises(ValidationError, match="deadline_s"):
        validate(scenario_data)


def test_validation_rechecks_unvalidated_model_copies(scenario_data):
    scenario = validate(scenario_data)
    invalid = scenario.model_copy(update={"nodes": ()})
    with pytest.raises(ValidationError, match="unknown"):
        validate(invalid)


def test_commands_roundtrip_through_discriminated_union():
    commands = (
        Place(request_id="r", stage_id="s", node_id="n"),
        Suspend(request_id="r", stage_id="s"),
        Resume(request_id="r", stage_id="s"),
        Reject(request_id="r", reason="capacity"),
        Defer(until_s=2),
    )
    adapter = TypeAdapter(tuple[Decision, ...])
    assert adapter.validate_json(adapter.dump_json(commands)) == commands
    with pytest.raises(ValidationError):
        adapter.validate_python([{"kind": "unknown"}])
    with pytest.raises(ValidationError):
        adapter.validate_python([{"kind": "place", "request_id": "r", "stage_id": "s"}])


def test_run_and_advance_contracts_serialize_without_backend(scenario_data):
    run = RunSpec(scenario=validate(scenario_data), external=True)
    assert RunSpec.model_validate_json(run.model_dump_json()) == run
    view = StateView(now_s=2, stages=(StageState(request_id="r", stage_id="s"),))
    decision = DecisionRequest(decision_id="d", time_s=2, view=view)
    for advance in (
        AdvanceResult(kind="decision", time_s=2, view=view, decision=decision),
        AdvanceResult(kind="time", time_s=2, view=view),
        AdvanceResult(
            kind="finished",
            time_s=2,
            view=view,
            result=RunResult(run_id="run", now_s=2, state=view),
        ),
    ):
        assert AdvanceResult.model_validate_json(advance.model_dump_json()) == advance
    with pytest.raises(ValidationError, match="decision payload"):
        AdvanceResult(kind="decision", time_s=2, view=view)
    with pytest.raises(ValidationError, match="result payload"):
        AdvanceResult(kind="finished", time_s=2, view=view)
    with pytest.raises(ValidationError, match="decision payload"):
        AdvanceResult(kind="time", time_s=2, view=view, decision=decision)
    assert "simgrid" not in json.dumps(RunSpec.model_json_schema()).lower()


def test_unknown_fields_fail_instead_of_silently_changing_semantics():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NodeSpec(id="n", speed_flops=1, memory_bytes=1, storage_bytes=1, core=2)


def test_artifact_state_accepts_scoped_outputs_without_relaxing_spec_ids():
    for key in ("global", "request/output"):
        state = ArtifactState(artifact_id=key, locations=("edge",))
        assert ArtifactState.model_validate_json(state.model_dump_json()) == state
        with pytest.raises(ValidationError, match="frozen"):
            state.artifact_id = "changed"
    for key in ("/output", "request/", "request/output/extra", "request/has space"):
        with pytest.raises(ValidationError):
            ArtifactState(artifact_id=key)
    with pytest.raises(ValidationError):
        ArtifactSpec(id="request/output", size_bytes=0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_cpu_speed_and_bandwidth_are_strictly_positive(value):
    with pytest.raises(ValidationError):
        NodeSpec(id="n", speed_flops=value, memory_bytes=0, storage_bytes=0)
    with pytest.raises(ValidationError):
        LinkSpec(id="l", bandwidth_bytes_s=value)


def test_initial_storage_counts_all_replicas_per_node(scenario_data):
    scenario_data["artifacts"].append(
        {"id": "replica", "size_bytes": 90, "locations": ["client", "edge"]}
    )
    validate(scenario_data)
    scenario_data["artifacts"][1]["size_bytes"] = 91
    with pytest.raises(ValidationError, match="storage capacity of node client"):
        validate(scenario_data)


def test_declared_outputs_require_explicit_matching_producer(scenario_data):
    del scenario_data["workflows"][0]["artifacts"][1]["producer_stage"]
    with pytest.raises(ValidationError, match="producer_stage must match"):
        validate(scenario_data)


def test_content_only_workflow_binds_every_artifact(scenario_data):
    workflow = scenario_data["workflows"][0]
    workflow["stages"] = []
    workflow["artifacts"] = [{"id": "input", "size_bytes": 10}]
    validate(scenario_data)
    workflow["artifacts"].append({"id": "second", "size_bytes": 10})
    with pytest.raises(ValidationError, match="exactly cover"):
        validate(scenario_data)
    scenario_data["requests"][0]["input_bindings"].append(
        {"artifact_id": "second", "global_artifact_id": "global"}
    )
    validate(scenario_data)


@pytest.mark.parametrize(
    "status",
    [
        "WAITING_DEPENDENCIES",
        "READY",
        "WAITING_DATA",
        "QUEUED",
        "RUNNING",
        "SUSPENDED",
        "SUCCEEDED",
        "CANCELLED",
        "FAILED",
    ],
)
def test_stage_lifecycle_uses_exact_uppercase_statuses(status):
    assert StageState(request_id="r", stage_id="s", status=status).status == status
    with pytest.raises(ValidationError):
        StageState(request_id="r", stage_id="s", status=status.lower())


@pytest.mark.parametrize(
    "status",
    [
        "PENDING",
        "ACTIVE",
        "SUCCEEDED",
        "REJECTED",
        "FAILED",
        "TIMED_OUT",
    ],
)
def test_request_lifecycle_uses_exact_uppercase_statuses(status):
    assert RequestState(request_id="r", status=status).status == status
    with pytest.raises(ValidationError):
        RequestState(request_id="r", status=status.lower())


@pytest.mark.parametrize("value", ["text", 2, 2.5, True, None])
def test_domain_event_details_preserve_scalar_types(value):
    event = DomainEvent(
        time_s=0,
        kind="stage_ready",
        entity_id="r/s",
        sequence=1,
        details=(Detail(name="value", value=value),),
    )
    restored = DomainEvent.model_validate_json(event.model_dump_json())
    assert type(restored.details[0].value) is type(value)
    assert restored == event
    with pytest.raises(ValidationError, match="frozen"):
        event.details[0].value = "changed"


@pytest.mark.parametrize("value", [{"nested": 1}, [1], (1,), float("nan"), float("inf")])
def test_domain_event_details_reject_containers_and_nonfinite_values(value):
    with pytest.raises(ValidationError):
        Detail(name="value", value=value)


def test_decision_candidates_are_scoped_and_immutable():
    decision = DecisionRequest(
        decision_id="d",
        time_s=0,
        view=StateView(),
        run_id="run",
        revision=3,
        candidates=(Candidate(request_id="r", stage_id="s", nodes=("n",)),),
    )
    assert DecisionRequest.model_validate_json(decision.model_dump_json()) == decision
    with pytest.raises(ValidationError, match="candidate stages"):
        DecisionRequest(
            decision_id="d", time_s=0, view=StateView(), candidates=decision.candidates * 2
        )
    with pytest.raises(ValidationError, match="candidate nodes"):
        Candidate(request_id="r", stage_id="s", nodes=("n", "n"))


def test_metrics_manifest_and_truncated_result_roundtrip():
    metrics = RunMetrics(
        arrived=2,
        completed=1,
        unfinished=1,
        events=10,
        cache_hits=1,
        transfers=2,
        request_latency=(Metric(name="r", value=3),),
        stage_durations=(Metric(name="r/s", value=2),),
        cpu_utilization=(Metric(name="n", value=0.5),),
        link_utilization=(Metric(name="l", value=1),),
    )
    manifest = RunManifest(
        run_id="run",
        seed=3,
        scenario_hash="abc",
        python_version="3.11",
        simgrid_version="4.1",
        policy="edf",
    )
    result = RunResult(
        run_id="run",
        seed=3,
        metrics=metrics,
        manifest=manifest,
        completed=False,
        end_reason="until_s",
        reason="time limit",
    )
    assert RunResult.model_validate_json(result.model_dump_json()) == result
    assert result.manifest.schema_version == "1"
    with pytest.raises(ValidationError, match="must match result"):
        RunResult(run_id="other", seed=3, manifest=manifest)


@pytest.mark.parametrize(
    "data",
    [
        {"arrived": -1},
        {"transfers": True},
        {"cpu_utilization": [{"name": "n", "value": 1.1}]},
        {"link_utilization": [{"name": "l", "value": -0.1}]},
        {"request_latency": [{"name": "r", "value": float("inf")}]},
        {"stage_durations": [{"name": "r/s", "value": 1}] * 2},
    ],
)
def test_invalid_metrics_fail(data):
    with pytest.raises(ValidationError):
        RunMetrics.model_validate(data)


def test_pause_resume_overheads_are_finite_nonnegative_seconds(scenario_data):
    run = RunSpec(scenario=validate(scenario_data))
    assert run.pause_overhead_s == run.resume_overhead_s == 0
    for field in ("pause_overhead_s", "resume_overhead_s"):
        for value in (-1, float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                RunSpec(scenario=run.scenario, **{field: value})
