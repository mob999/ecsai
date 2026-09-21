"""Content-service contracts: physical queues and control, with no learning types."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from . import DTO, Bytes, Count, Identifier, NonNegativeFloat, PositiveFloat, Seconds


class TransferPoolSpec(DTO):
    max_active: Annotated[int, Field(ge=1, strict=True)] | None = None
    max_waiting: Count | None = None


class CacheNodeSpec(DTO):
    node_id: Identifier
    cluster_id: Identifier
    backhaul_link: Identifier
    delivery_link: Identifier
    total_bandwidth_bytes_s: PositiveFloat | None = None
    backhaul: TransferPoolSpec = Field(default_factory=TransferPoolSpec)
    delivery: TransferPoolSpec = Field(default_factory=TransferPoolSpec)


class SchedulerSpec(DTO):
    id: Identifier
    max_waiting: Count = 100
    service_s: PositiveFloat = 0.001


class ContentRequest(DTO):
    id: Identifier
    artifact_id: Identifier
    cluster_id: Identifier
    receiver: Identifier
    arrival_s: Seconds
    deadline_s: Seconds

    @model_validator(mode="after")
    def times(self) -> Self:
        if self.deadline_s < self.arrival_s:
            raise ValueError("deadline precedes arrival")
        return self


class ContentServiceSpec(DTO):
    origin: Identifier
    report_arrivals: bool = False
    max_retries: Count = 0
    retry_delay_s: Seconds = 0.1
    bandwidth_mode: Literal["independent", "shared"] = "independent"
    coalesce_backhaul: bool = True
    scheduler_release: Literal["continuous", "window"] = "continuous"
    schedulers: tuple[SchedulerSpec, ...]
    caches: tuple[CacheNodeSpec, ...]
    requests: tuple[ContentRequest, ...]
    size_scale_bytes: PositiveFloat = 7_500_000
    deadline_scale_s: PositiveFloat = 1.0

    @model_validator(mode="after")
    def references(self) -> Self:
        for values in (
            [s.id for s in self.schedulers],
            [c.node_id for c in self.caches],
            [r.id for r in self.requests],
        ):
            if len(values) != len(set(values)):
                raise ValueError("duplicate content-service IDs")
        if self.bandwidth_mode == "shared":
            if any(c.total_bandwidth_bytes_s is None for c in self.caches):
                raise ValueError("shared bandwidth requires a total capacity per cache")
            links = [link for c in self.caches for link in (c.backhaul_link, c.delivery_link)]
            if len(set(links)) != len(links):
                raise ValueError("shared bandwidth requires dedicated links per cache")
        schedulers = {s.id for s in self.schedulers}
        if not schedulers or not self.caches:
            raise ValueError("content service needs schedulers and caches")
        if any(c.cluster_id not in schedulers for c in self.caches) or any(
            r.cluster_id not in schedulers for r in self.requests
        ):
            raise ValueError("unknown scheduler")
        if any(not any(c.cluster_id == s for c in self.caches) for s in schedulers):
            raise ValueError("every scheduler needs a cache")
        return self


class SchedulerControl(DTO):
    # Request IDs bind decisions to a boundary snapshot, never to future arrivals.
    request_decisions: dict[Identifier, bool] = Field(default_factory=dict)
    backhaul_ratio: Annotated[float, Field(ge=0.05, le=0.95, allow_inf_nan=False)] = 0.5
    cluster_id: Identifier
    weights: tuple[
        Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)],
        Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)],
        Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)],
        Annotated[float, Field(ge=-5, le=5, allow_inf_nan=False)],
    ] = (0, 0, 0, -5)


class WindowControl(DTO):
    schedulers: tuple[SchedulerControl, ...]
    policy: Literal["threshold", "local", "forward", "random", "direct"] = "threshold"

    @model_validator(mode="after")
    def unique(self) -> Self:
        ids = [s.cluster_id for s in self.schedulers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scheduler controls")
        return self


class Serve(DTO):
    """A content-service decision made by the in-worker controller."""

    request_id: Identifier
    cache_node: Identifier


class PoolState(DTO):
    node_id: Identifier
    kind: Literal["backhaul", "delivery"]
    active: Count
    waiting: Count
    max_active: Count | None
    max_waiting: Count | None
    remaining_bytes: NonNegativeFloat = 0


class SchedulerState(DTO):
    cluster_id: Identifier
    waiting: tuple[Identifier, ...]
    active_request: Identifier | None = None
    capacity: Count


class ContentRequestState(DTO):
    request_id: Identifier
    original_request_id: Identifier | None = None
    attempt_index: Count = 0
    first_arrival_s: Seconds | None = None
    origin_cluster: Identifier
    cluster_id: Identifier
    artifact_id: Identifier
    size_bytes: Bytes
    arrival_s: Seconds
    deadline_s: Seconds
    status: str
    cache_node: Identifier | None = None
    forwarded: bool = False
    completed_s: Seconds | None = None
    reason: str | None = None


class ContentTransferState(DTO):
    transfer_id: Identifier
    artifact_id: Identifier
    node_id: Identifier
    kind: Literal["backhaul", "delivery"]
    status: Literal["waiting", "active"]
    request_ids: tuple[Identifier, ...]
    src: Identifier
    dst: Identifier
    size_bytes: Bytes
    remaining_bytes: NonNegativeFloat


class LinkCounter(DTO):
    link_id: Identifier
    bytes_sent: NonNegativeFloat
    capacity_byte_seconds: NonNegativeFloat


class ContentArrivalState(DTO):
    request_id: Identifier
    cluster_id: Identifier
    arrival_s: Seconds
    size_bytes: Bytes


class ContentView(DTO):
    # Scheduling snapshots retain all counters/pools but include only queued
    # scheduler requests and omit transfer details. Window-release snapshots also
    # include the active scheduler request for fixed-slot control. inspect() remains full.
    scope: Literal["full", "scheduling"] = "full"
    now_s: Seconds
    schedulers: tuple[SchedulerState, ...] = ()
    pools: tuple[PoolState, ...] = ()
    requests: tuple[ContentRequestState, ...] = ()
    transfers: tuple[ContentTransferState, ...] = ()
    links: tuple[LinkCounter, ...] = ()
    arrived: Count = 0
    completed: Count = 0
    timed_out: Count = 0
    rejected: Count = 0
    cache_hits: Count = 0
    cache_lookups: Count = 0
    forwarded: Count = 0
    overflows: Count = 0
    cancelled_transfers: Count = 0
    latencies_s: tuple[Seconds, ...] = ()
    # Retry-enabled runs expose completed attempts, including failures, per window.
    attempt_outcomes: tuple[ContentRequestState, ...] = ()
    # Actual original/retry arrivals in the last control window, never future requests.
    window_arrivals: tuple[ContentArrivalState, ...] = ()


class WindowResult(DTO):
    """Compact window response: live requests plus newly terminal requests, not full history."""

    kind: Literal["time", "finished"]
    view: ContentView
    start_s: Seconds
    completed: Count = 0
    timed_out: Count = 0
    rejected: Count = 0
    link_bytes: tuple[LinkCounter, ...] = ()
    simulation_wall_s: NonNegativeFloat = 0
