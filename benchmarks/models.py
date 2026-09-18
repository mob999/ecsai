"""Pydantic benchmark settings and persisted measurement contracts."""

from pathlib import Path
from typing import Annotated, Literal

from edge_sim_models import RunMetrics
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

Count = Annotated[int, Field(ge=0)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Workload = Literal["compute", "network", "mixed"]


class BenchmarkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDGE_BENCH_", extra="forbid", frozen=True)

    nodes: tuple[Literal[32, 128], ...] = (32, 128)
    workloads: tuple[Workload, ...] = ("compute", "network", "mixed")
    workers: tuple[Literal[1, 2, 4, 8], ...] = (1, 2, 4, 8)
    repeats: Annotated[int, Field(ge=1)] = 2
    ipc_samples: Annotated[int, Field(ge=1)] = 5
    trace: bool = False
    output: Path = Path("benchmarks/results.json")
    baseline: Path | None = None
    allow_regression: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None
    ) = None


class Measurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunMeasurement(Measurement):
    run_id: str
    success: bool
    error: str | None = None
    startup_s: Duration | None = None
    inspect_rpc_s: tuple[Duration, ...] = ()
    simulated_s: Duration | None = None
    request_count: Count | None = None
    completed_requests: Count | None = None
    stage_count: Count | None = None
    completed_stages: Count | None = None
    advance_calls: Count | None = None
    batch_advance_wall_s: Duration | None = None
    event_count: Count | None = None
    retained_event_count: Count | None = None
    metrics: RunMetrics | None = None


class CellMeasurement(Measurement):
    nodes: Literal[32, 128]
    workload: Workload
    workers: Literal[1, 2, 4, 8]
    requested_runs: Count
    successful_runs: Count
    failed_runs: Count
    requests_per_run: Count
    completed_requests: Count
    wall_s: Duration
    successful_runs_per_wall_s: Duration
    completed_requests_per_wall_s: Duration
    startup_median_s: Duration | None
    inspect_rpc_median_s: Duration | None
    inspect_rpc_p95_s: Duration | None
    rss_sample_count: Count
    rss_samples_with_workers: Count
    rss_max_observed_workers: Count
    rss_sampling_interval_s: Duration
    parent_peak_sampled_rss_bytes: Count | None
    workers_peak_sampled_rss_bytes: Count | None
    parent_plus_workers_peak_sampled_rss_bytes: Count | None
    rss_sampling_errors: tuple[str, ...]
    runs: tuple[RunMeasurement, ...]


class Environment(Measurement):
    platform: str
    python: str
    cpu_count: Count | None
    simgrid: str
    trace: bool


class Methodology(Measurement):
    concurrency: str
    startup: str
    ipc: str
    rss: str
    throughput: str
    warmup: str


class BenchmarkReport(Measurement):
    schema_version: Literal[1] = 1
    environment: Environment
    methodology: Methodology
    cells: tuple[CellMeasurement, ...] = ()
    comparison: "Comparison | None" = None


class CellComparison(Measurement):
    nodes: Literal[32, 128]
    workload: Workload
    workers: Literal[1, 2, 4, 8]
    throughput_change_percent: float
    rss_change_percent: float
    regressions: tuple[str, ...] = ()


class Comparison(Measurement):
    status: Literal["passed", "regressed", "allowed", "invalid"]
    baseline: str
    explanation: str | None = None
    errors: tuple[str, ...] = ()
    cells: tuple[CellComparison, ...] = ()


BenchmarkReport.model_rebuild()
