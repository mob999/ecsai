"""Guardrails for comparisons; timings themselves are not performance assertions."""

import pytest
from pydantic import ValidationError

from benchmarks.compare import compare_reports
from benchmarks.models import (
    BenchmarkReport,
    BenchmarkSettings,
    CellMeasurement,
    Environment,
    Methodology,
)


def report(throughput=100.0, rss=1000):
    cell = CellMeasurement(
        nodes=32,
        workload="compute",
        workers=1,
        requested_runs=1,
        successful_runs=1,
        failed_runs=0,
        requests_per_run=31,
        completed_requests=31,
        wall_s=1,
        successful_runs_per_wall_s=1,
        completed_requests_per_wall_s=throughput,
        startup_median_s=0.1,
        inspect_rpc_median_s=0.001,
        inspect_rpc_p95_s=0.002,
        rss_sample_count=2,
        rss_samples_with_workers=1,
        rss_max_observed_workers=1,
        rss_sampling_interval_s=0.02,
        parent_peak_sampled_rss_bytes=500,
        workers_peak_sampled_rss_bytes=500,
        parent_plus_workers_peak_sampled_rss_bytes=rss,
        rss_sampling_errors=(),
        runs=(),
    )
    return BenchmarkReport(
        environment=Environment(
            platform="test-host", python="3.11", cpu_count=8, simgrid="4.1", trace=False
        ),
        methodology=Methodology(
            concurrency="batch",
            startup="ready",
            ipc="inspect",
            rss="sampled",
            throughput="wall",
            warmup="none",
        ),
        cells=(cell,),
    )


@pytest.mark.parametrize(
    "throughput,rss,status",
    [
        (100, 1000, "passed"),
        (85, 1200, "passed"),
        (84.99, 1000, "regressed"),
        (100, 1201, "regressed"),
    ],
)
def test_strict_regression_thresholds(throughput, rss, status):
    assert compare_reports(report(throughput, rss), report(), "base.json").status == status


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform", "other-host"),
        ("python", "3.12"),
        ("cpu_count", 16),
        ("trace", True),
        ("simgrid", "other"),
    ],
)
def test_environment_mismatch_cannot_be_waived(field, value):
    current = report()
    current = current.model_copy(
        update={
            "environment": current.environment.model_copy(update={field: value}),
        }
    )
    result = compare_reports(current, report(), "base.json", "intentional change")
    assert result.status == "invalid"
    assert field in result.errors[0]


def test_explicit_explanation_retains_regression_evidence():
    result = compare_reports(report(80, 1300), report(), "base.json", "extra tracing cost")
    assert result.status == "allowed"
    assert result.explanation == "extra tracing cost"
    assert len(result.cells[0].regressions) == 2
    assert compare_reports(report(80), report(), "base.json", " ").status == "regressed"


def test_missing_cell_and_missing_rss_are_invalid():
    current = report()
    empty = current.model_copy(update={"cells": ()})
    assert compare_reports(current, empty, "base.json").status == "invalid"
    assert compare_reports(report(rss=None), report(), "base.json").status == "invalid"


def test_blank_override_is_not_valid_settings():
    with pytest.raises(ValidationError):
        BenchmarkSettings(allow_regression=" ")


def test_failed_runs_cannot_be_waived():
    current = report()
    current = current.model_copy(
        update={
            "cells": (current.cells[0].model_copy(update={"failed_runs": 1}),),
        }
    )
    assert compare_reports(current, report(), "base.json", "accepted").status == "invalid"
