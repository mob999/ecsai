"""Compare matching measurements, refusing incompatible or missing evidence."""

from .models import BenchmarkReport, CellComparison, Comparison


def compare_reports(
    current: BenchmarkReport,
    baseline: BenchmarkReport,
    baseline_path: str,
    explanation: str | None = None,
) -> Comparison:
    errors = []
    for field in ("platform", "python", "cpu_count", "simgrid", "trace"):
        before = getattr(baseline.environment, field)
        after = getattr(current.environment, field)
        if before != after or (field == "cpu_count" and before is None):
            errors.append(f"environment.{field} differs or is unavailable: {before!r} -> {after!r}")
    if errors:
        return Comparison(status="invalid", baseline=baseline_path, errors=tuple(errors))
    indexed = {(c.nodes, c.workload, c.workers): c for c in baseline.cells}
    if len(indexed) != len(baseline.cells) or not current.cells:
        errors.append("Baseline has duplicate cells or current report has no cells")
    seen = set()
    comparisons = []
    for cell in current.cells:
        key = (cell.nodes, cell.workload, cell.workers)
        if key in seen:
            errors.append(f"Duplicate current cell: {key}")
            continue
        seen.add(key)
        previous = indexed.get(key)
        if previous is None:
            errors.append(f"Missing baseline cell: {key}")
            continue
        if (
            cell.failed_runs
            or previous.failed_runs
            or cell.successful_runs != cell.requested_runs
            or previous.successful_runs != previous.requested_runs
        ):
            errors.append(f"Failed or incomplete runs in cell: {key}")
            continue
        if (cell.requested_runs, cell.requests_per_run) != (
            previous.requested_runs,
            previous.requests_per_run,
        ):
            errors.append(f"Run/request counts differ in cell: {key}")
            continue
        rss = cell.parent_plus_workers_peak_sampled_rss_bytes
        previous_rss = previous.parent_plus_workers_peak_sampled_rss_bytes
        throughput = previous.completed_requests_per_wall_s
        if (
            rss is None
            or previous_rss is None
            or previous_rss <= 0
            or throughput <= 0
            or not cell.rss_samples_with_workers
            or not previous.rss_samples_with_workers
            or cell.rss_sampling_errors
            or previous.rss_sampling_errors
        ):
            errors.append(f"Missing/invalid throughput or RSS evidence in cell: {key}")
            continue
        throughput_change = 100 * (cell.completed_requests_per_wall_s / throughput - 1)
        rss_change = 100 * (rss / previous_rss - 1)
        regressions = []
        if cell.completed_requests_per_wall_s < throughput * 0.85:
            regressions.append("throughput decreased by more than 15%")
        if rss > previous_rss * 1.20:
            regressions.append("parent-plus-worker RSS increased by more than 20%")
        comparisons.append(
            CellComparison(
                nodes=cell.nodes,
                workload=cell.workload,
                workers=cell.workers,
                throughput_change_percent=throughput_change,
                rss_change_percent=rss_change,
                regressions=tuple(regressions),
            )
        )
    regressed = any(c.regressions for c in comparisons)
    explanation = explanation.strip() if explanation else None
    status = (
        "invalid"
        if errors
        else ("allowed" if regressed and explanation else "regressed" if regressed else "passed")
    )
    return Comparison(
        status=status,
        baseline=baseline_path,
        explanation=explanation,
        errors=tuple(errors),
        cells=tuple(comparisons),
    )
