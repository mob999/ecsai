"""Immutable, backend-independent contracts for edge simulation.

All times are seconds, capacities are bytes, and work is measured in FLOPs.
Routes are directed; sharing a link ID expresses contention on that link.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier: TypeAlias = Annotated[
    str, StringConstraints(strict=True, min_length=1, pattern=r"^[^\s/]+$")
]
StageKey: TypeAlias = Annotated[str, StringConstraints(strict=True, pattern=r"^[^\s/]+/[^\s/]+$")]
NodeId: TypeAlias = Identifier
LinkId: TypeAlias = Identifier
ArtifactId: TypeAlias = Identifier
WorkflowId: TypeAlias = Identifier
StageId: TypeAlias = Identifier
RequestId: TypeAlias = Identifier
RunId: TypeAlias = Identifier
NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat: TypeAlias = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Count: TypeAlias = Annotated[int, Field(ge=0, strict=True)]
Utilization: TypeAlias = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Scalar: TypeAlias = (
    Annotated[str, Field(strict=True)]
    | Annotated[int, Field(strict=True)]
    | Annotated[float, Field(strict=True, allow_inf_nan=False)]
    | Annotated[bool, Field(strict=True)]
    | None
)
Seconds: TypeAlias = NonNegativeFloat
Flops: TypeAlias = NonNegativeFloat
Bytes: TypeAlias = Annotated[int, Field(ge=0, strict=True)]
NodeRole: TypeAlias = Literal["client", "edge", "cloud"]
PolicyName: TypeAlias = Literal["fifo", "edf", "round_robin"]
StageStatus: TypeAlias = Literal[
    "WAITING_DEPENDENCIES",
    "READY",
    "WAITING_DATA",
    "QUEUED",
    "RUNNING",
    "SUSPENDED",
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
]
RequestStatus: TypeAlias = Literal[
    "PENDING", "ACTIVE", "SUCCEEDED", "REJECTED", "FAILED", "TIMED_OUT"
]


class DTO(BaseModel):
    """A frozen value object with a closed schema and immutable nested values."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, revalidate_instances="always"
    )


def _unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


def _references(values: tuple[str, ...], known: set[str], label: str) -> None:
    missing = set(values) - known
    if missing:
        raise ValueError(f"unknown {label}: {sorted(missing)}")


class NodeSpec(DTO):
    id: NodeId
    role: NodeRole = "edge"
    cores: Annotated[int, Field(ge=1, strict=True)] = 1
    speed_flops: PositiveFloat
    memory_bytes: Bytes
    storage_bytes: Bytes


class LinkSpec(DTO):
    id: LinkId
    bandwidth_bytes_s: PositiveFloat
    latency_s: Seconds = 0


class RouteSpec(DTO):
    src: NodeId
    dst: NodeId
    links: tuple[LinkId, ...] = ()

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _unique(self.links, "route links")
        if self.src != self.dst and not self.links:
            raise ValueError("a route between distinct nodes requires links")
        return self


class ArtifactSpec(DTO):
    """An initially available global object; locations identify replicas."""

    id: ArtifactId
    size_bytes: Bytes
    locations: tuple[NodeId, ...] = ()
    cacheable: bool = True

    @model_validator(mode="after")
    def validate_locations(self) -> Self:
        _unique(self.locations, "artifact locations")
        return self


class WorkflowArtifactSpec(DTO):
    id: ArtifactId
    size_bytes: Bytes
    producer_stage: StageId | None = None
    cacheable: bool = True


class StageSpec(DTO):
    id: StageId
    flops: Flops
    memory_bytes: Bytes = 0
    inputs: tuple[ArtifactId, ...] = ()
    outputs: tuple[ArtifactId, ...] = ()
    depends_on: tuple[StageId, ...] = ()
    eligible_nodes: tuple[NodeId, ...] = ()

    @model_validator(mode="after")
    def validate_lists(self) -> Self:
        for name in ("inputs", "outputs", "depends_on", "eligible_nodes"):
            _unique(getattr(self, name), f"stage {name}")
        if self.id in self.depends_on:
            raise ValueError("stage cannot depend on itself")
        if set(self.inputs) & set(self.outputs):
            raise ValueError("stage cannot consume its own output")
        return self


class WorkflowSpec(DTO):
    id: WorkflowId
    stages: tuple[StageSpec, ...]
    artifacts: tuple[WorkflowArtifactSpec, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        _unique(tuple(s.id for s in self.stages), "stage IDs")
        _unique(tuple(a.id for a in self.artifacts), "workflow artifact IDs")
        stage_ids = {s.id for s in self.stages}
        artifact_ids = {a.id for a in self.artifacts}
        producers: dict[str, str] = {}
        for stage in self.stages:
            _references(stage.depends_on, stage_ids, "dependency stages")
            _references(stage.inputs + stage.outputs, artifact_ids, "workflow artifacts")
            for output in stage.outputs:
                if output in producers:
                    raise ValueError(f"multiple producers for artifact {output}")
                producers[output] = stage.id
        for artifact in self.artifacts:
            if artifact.producer_stage is not None:
                _references((artifact.producer_stage,), stage_ids, "producer stages")
            if producers.get(artifact.id) != artifact.producer_stage:
                raise ValueError(
                    f"producer must declare artifact {artifact.id} as an output "
                    "and producer_stage must match"
                )

        # Data dependencies count as graph edges even without explicit depends_on.
        dependencies = {
            s.id: set(s.depends_on) | {producers[a] for a in s.inputs if a in producers}
            for s in self.stages
        }
        dependents: dict[str, list[str]] = {sid: [] for sid in stage_ids}
        for sid, parents in dependencies.items():
            for parent in parents:
                dependents[parent].append(sid)
        ready = [sid for sid, parents in dependencies.items() if not parents]
        visited = 0
        while ready:
            parent = ready.pop()
            visited += 1
            for child in dependents[parent]:
                dependencies[child].remove(parent)
                if not dependencies[child]:
                    ready.append(child)
        if visited != len(stage_ids):
            raise ValueError("workflow dependency cycle")
        return self


class InputBinding(DTO):
    artifact_id: ArtifactId
    global_artifact_id: ArtifactId


class RequestSpec(DTO):
    id: RequestId
    workflow: WorkflowId
    arrival_s: Seconds = 0
    deadline_s: Seconds | None = None
    receiver: NodeId
    input_bindings: tuple[InputBinding, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _unique(tuple(b.artifact_id for b in self.input_bindings), "input bindings")
        if self.deadline_s is not None and self.deadline_s < self.arrival_s:
            raise ValueError("deadline_s must be at or after arrival_s")
        return self


class ScenarioSpec(DTO):
    schema_version: Literal["1"] = "1"
    cpu_model: Literal["Cas01"] = "Cas01"
    network_model: Literal["raw"] = "raw"
    nodes: tuple[NodeSpec, ...]
    links: tuple[LinkSpec, ...] = ()
    routes: tuple[RouteSpec, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = ()
    workflows: tuple[WorkflowSpec, ...] = ()
    requests: tuple[RequestSpec, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for name in ("nodes", "links", "artifacts", "workflows", "requests"):
            _unique(tuple(item.id for item in getattr(self, name)), f"{name} IDs")
        nodes = {n.id for n in self.nodes}
        links = {link.id for link in self.links}
        artifacts = {a.id: a for a in self.artifacts}
        workflows = {w.id: w for w in self.workflows}
        pairs: set[tuple[str, str]] = set()
        for route in self.routes:
            _references((route.src, route.dst), nodes, "route nodes")
            _references(route.links, links, "route links")
            if (route.src, route.dst) in pairs:
                raise ValueError(f"duplicate route {route.src} -> {route.dst}")
            pairs.add((route.src, route.dst))
        storage_used = dict.fromkeys(nodes, 0)
        for artifact in self.artifacts:
            _references(artifact.locations, nodes, "artifact locations")
            for location in artifact.locations:
                storage_used[location] += artifact.size_bytes
        for node in self.nodes:
            if storage_used[node.id] > node.storage_bytes:
                raise ValueError(f"initial artifacts exceed storage capacity of node {node.id}")
        for workflow in self.workflows:
            for stage in workflow.stages:
                _references(stage.eligible_nodes, nodes, "eligible nodes")
        for request in self.requests:
            _references((request.workflow,), set(workflows), "request workflows")
            _references((request.receiver,), nodes, "request receivers")
            workflow = workflows[request.workflow]
            local_artifacts = {a.id: a for a in workflow.artifacts}
            produced = {a for s in workflow.stages for a in s.outputs}
            external = {a for s in workflow.stages for a in s.inputs} - produced
            if not workflow.stages:
                external = set(local_artifacts) - produced
            bound = {b.artifact_id for b in request.input_bindings}
            if bound != external:
                raise ValueError("input bindings must exactly cover workflow external inputs")
            for binding in request.input_bindings:
                _references((binding.global_artifact_id,), set(artifacts), "global artifacts")
                source = artifacts[binding.global_artifact_id]
                if source.size_bytes != local_artifacts[binding.artifact_id].size_bytes:
                    raise ValueError("bound artifact sizes must match")
                if not source.locations:
                    raise ValueError("bound global artifact must have an initial location")
        return self


class PluginSpec(DTO):
    role: Literal["admission", "placement", "replica", "scheduling", "preemption", "cache"]
    factory: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")]


class PolicySpec(DTO):
    name: PolicyName = "fifo"
    external: bool = False
    plugins: tuple[PluginSpec, ...] = ()

    @model_validator(mode="after")
    def validate_plugins(self) -> Self:
        _unique(tuple(p.role for p in self.plugins), "policy plugin roles")
        return self


class RunSpec(DTO):
    scenario: ScenarioSpec
    seed: Annotated[int, Field(ge=0, strict=True)] = 0
    run_id: RunId = "run"
    until_s: Seconds | None = None
    trace: bool = True
    policy: PolicySpec = Field(default_factory=PolicySpec)
    external: bool = False
    pause_overhead_s: Seconds = 0
    resume_overhead_s: Seconds = 0


class NodeState(DTO):
    node_id: NodeId
    cores_used: Annotated[int, Field(ge=0, strict=True)] = 0
    memory_used_bytes: Bytes = 0
    storage_used_bytes: Bytes = 0


class ArtifactState(DTO):
    artifact_id: ArtifactId | StageKey
    request_id: RequestId | None = None
    size_bytes: Bytes = 0
    locations: tuple[NodeId, ...] = ()
    cacheable: bool = True


class StageState(DTO):
    request_id: RequestId
    stage_id: StageId
    status: StageStatus = "WAITING_DEPENDENCIES"
    node_id: NodeId | None = None
    remaining_flops: Flops = 0
    started_s: Seconds | None = None
    completed_s: Seconds | None = None
    reason: str | None = None
    durations: tuple[Metric, ...] = ()

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if (
            self.started_s is not None
            and self.completed_s is not None
            and self.completed_s < self.started_s
        ):
            raise ValueError("completed_s must be at or after started_s")
        return self


class RequestState(DTO):
    request_id: RequestId
    status: RequestStatus = "PENDING"
    arrival_s: Seconds = 0
    deadline_s: Seconds | None = None
    completed_s: Seconds | None = None
    reason: str | None = None


class StateView(DTO):
    now_s: Seconds = 0
    nodes: tuple[NodeState, ...] = ()
    artifacts: tuple[ArtifactState, ...] = ()
    stages: tuple[StageState, ...] = ()
    requests: tuple[RequestState, ...] = ()


class Place(DTO):
    kind: Literal["place"] = "place"
    request_id: RequestId
    stage_id: StageId
    node_id: NodeId


class Suspend(DTO):
    kind: Literal["suspend"] = "suspend"
    request_id: RequestId
    stage_id: StageId


class Resume(DTO):
    kind: Literal["resume"] = "resume"
    request_id: RequestId
    stage_id: StageId


class Reject(DTO):
    kind: Literal["reject"] = "reject"
    request_id: RequestId
    reason: Annotated[str, StringConstraints(min_length=1)]


class Defer(DTO):
    kind: Literal["defer"] = "defer"
    until_s: Seconds


Decision: TypeAlias = Annotated[
    Place | Suspend | Resume | Reject | Defer, Field(discriminator="kind")
]
DecisionCommand: TypeAlias = Decision


class Detail(DTO):
    name: Identifier
    value: Scalar


class Metric(DTO):
    name: Annotated[str, StringConstraints(strict=True, min_length=1)]
    value: NonNegativeFloat


class Candidate(DTO):
    request_id: RequestId
    stage_id: StageId
    nodes: tuple[NodeId, ...] = ()

    @model_validator(mode="after")
    def validate_nodes(self) -> Self:
        _unique(self.nodes, "candidate nodes")
        return self


class DomainEvent(DTO):
    """Structured trace entry; optional fields avoid mutable metadata dictionaries."""

    time_s: Seconds
    kind: Identifier
    sequence: Count = 0
    entity_id: Annotated[str, StringConstraints(strict=True, min_length=1)] | None = None
    details: tuple[Detail, ...] = ()
    request_id: RequestId | None = None
    stage_id: StageId | None = None
    node_id: NodeId | None = None
    artifact_id: ArtifactId | None = None
    src: NodeId | None = None
    dst: NodeId | None = None
    size_bytes: Bytes | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_details(self) -> Self:
        _unique(tuple(detail.name for detail in self.details), "event detail names")
        return self


Event = DomainEvent


class RunMetrics(DTO):
    """Durations and latency use seconds; utilization is a fraction in [0, 1]."""

    arrived: Count = 0
    completed: Count = 0
    rejected: Count = 0
    failed: Count = 0
    timed_out: Count = 0
    unfinished: Count = 0
    cache_hits: Count = 0
    transfers: Count = 0
    cpu_utilization: tuple[Metric, ...] = ()
    link_utilization: tuple[Metric, ...] = ()
    request_latency: tuple[Metric, ...] = ()
    stage_durations: tuple[Metric, ...] = ()
    events: Count = 0

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        for field in ("cpu_utilization", "link_utilization", "request_latency", "stage_durations"):
            _unique(tuple(metric.name for metric in getattr(self, field)), f"{field} names")
        for metric in self.cpu_utilization + self.link_utilization:
            if metric.value > 1:
                raise ValueError("utilization must be between 0 and 1")
        return self


class RunManifest(DTO):
    run_id: RunId
    seed: Count = 0
    policy: PolicyName = "fifo"
    cpu_model: Literal["Cas01"] = "Cas01"
    network_model: Literal["raw"] = "raw"
    scenario_hash: str | None = None
    python_version: str = "unspecified"
    simgrid_version: str | None = None
    framework_version: str = "0.1.0"
    schema_version: str = "1"


class RunResult(DTO):
    run_id: RunId
    seed: Annotated[int, Field(ge=0, strict=True)] = 0
    now_s: Seconds = 0
    state: StateView = Field(default_factory=StateView)
    events: tuple[Event, ...] = ()
    completed: bool = True
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    manifest: RunManifest | None = None
    end_reason: str = "completed"
    reason: str | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.manifest is not None and (
            self.manifest.run_id != self.run_id or self.manifest.seed != self.seed
        ):
            raise ValueError("manifest run_id and seed must match result")
        return self


class DecisionRequest(DTO):
    decision_id: Identifier
    time_s: Seconds
    view: StateView
    reason: str = "ready"
    run_id: RunId = "run"
    revision: Count = 0
    candidates: tuple[Candidate, ...] = ()

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        _unique(
            tuple(f"{candidate.request_id}/{candidate.stage_id}" for candidate in self.candidates),
            "candidate stages",
        )
        return self


class AdvanceResult(DTO):
    kind: Literal["decision", "time", "finished"]
    time_s: Seconds
    view: StateView
    decision: DecisionRequest | None = None
    result: RunResult | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if (self.kind == "decision") != (self.decision is not None):
            raise ValueError("decision payload is required exactly for decision advances")
        if (self.kind == "finished") != (self.result is not None):
            raise ValueError("result payload is required exactly for finished advances")
        return self


def validate(scenario: ScenarioSpec | dict) -> ScenarioSpec:
    """Parse and validate a scenario; raises pydantic.ValidationError on invalid input."""
    return ScenarioSpec.model_validate(scenario)


__all__ = [
    "AdvanceResult",
    "ArtifactId",
    "ArtifactSpec",
    "ArtifactState",
    "Bytes",
    "Candidate",
    "Count",
    "DTO",
    "Decision",
    "DecisionCommand",
    "DecisionRequest",
    "Defer",
    "Detail",
    "DomainEvent",
    "Event",
    "Flops",
    "Identifier",
    "InputBinding",
    "LinkId",
    "LinkSpec",
    "Metric",
    "NodeId",
    "NodeRole",
    "NodeSpec",
    "NodeState",
    "NonNegativeFloat",
    "Place",
    "PolicyName",
    "PluginSpec",
    "PolicySpec",
    "PositiveFloat",
    "Reject",
    "RequestId",
    "RequestSpec",
    "RequestState",
    "RequestStatus",
    "Resume",
    "RouteSpec",
    "RunId",
    "RunManifest",
    "RunMetrics",
    "RunResult",
    "RunSpec",
    "ScenarioSpec",
    "Scalar",
    "Seconds",
    "StageId",
    "StageKey",
    "StageSpec",
    "StageState",
    "StageStatus",
    "StateView",
    "Suspend",
    "Utilization",
    "WorkflowArtifactSpec",
    "WorkflowId",
    "WorkflowSpec",
    "validate",
]
