"""Measure independent spawned sessions; never fabricate unavailable metrics."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import threading
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from edge_sim import BatchRunner, SDKError
from edge_sim_models import PolicySpec, RunSpec

from examples.scenarios import benchmark_scenario

from .compare import compare_reports
from .models import BenchmarkReport, BenchmarkSettings, CellMeasurement


class RssSampler:
    """Sample parent and explicitly registered SDK worker RSS using macOS/Linux ps."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.pids: set[int] = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.samples: list[tuple[int, int, int]] = []
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def register(self, pid: int) -> None:
        with self.lock:
            self.pids.add(pid)

    def unregister(self, pid: int) -> None:
        with self.lock:
            self.pids.discard(pid)

    def _sample(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                pids = {os.getpid(), *self.pids}
            try:
                output = subprocess.run(
                    ["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, sorted(pids)))],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout
                rows = {
                    int(pid): int(rss) * 1024
                    for pid, rss in (line.split() for line in output.splitlines() if line.strip())
                }
                if os.getpid() not in rows:
                    raise RuntimeError("ps did not return parent RSS")
                self.samples.append(
                    (
                        rows[os.getpid()],
                        sum(rss for pid, rss in rows.items() if pid != os.getpid()),
                        len(rows) - 1,
                    )
                )
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(self.interval_s)

    def summary(self) -> dict:
        worker_samples = [(p, w) for p, w, count in self.samples if count]
        return {
            "rss_sample_count": len(self.samples),
            "rss_samples_with_workers": len(worker_samples),
            "rss_max_observed_workers": max((n for _, _, n in self.samples), default=0),
            "rss_sampling_interval_s": self.interval_s,
            "parent_peak_sampled_rss_bytes": max((p for p, _, _ in self.samples), default=None),
            "workers_peak_sampled_rss_bytes": max((w for _, w in worker_samples), default=None),
            "parent_plus_workers_peak_sampled_rss_bytes": max(
                (p + w for p, w, _ in self.samples),
                default=None,
            ),
            "rss_sampling_errors": self.errors,
        }


def measure_wave(runs: list[RunSpec], sampler: RssSampler, ipc_samples: int) -> list[dict]:
    measurements = {run.run_id: {"run_id": run.run_id, "success": False} for run in runs}
    pids = []
    try:
        with BatchRunner(workers=len(runs)) as runner:
            for run in runs:
                measurement = measurements[run.run_id]
                before = perf_counter()
                runner.submit(run)
                session = runner.session(run.run_id)
                measurement["startup_s"] = perf_counter() - before
                if session.pid is not None:
                    sampler.register(session.pid)
                    pids.append(session.pid)
                latencies = []
                for _ in range(ipc_samples):
                    before = perf_counter()
                    session.inspect()
                    latencies.append(perf_counter() - before)
                measurement["inspect_rpc_s"] = latencies
            before = perf_counter()
            responses = runner.advance_all()
            advance_wall_s = perf_counter() - before
            for run in runs:
                measurement = measurements[run.run_id]
                response = responses.get(run.run_id)
                if isinstance(response, SDKError):
                    measurement["error"] = str(response)
                    continue
                if response is None or response.kind != "finished":
                    measurement["error"] = "Built-in policy did not finish after advance_all()"
                    continue
                result = runner.result(run.run_id)
                requests = result.state.requests
                stages = result.state.stages
                completed = sum(r.status == "SUCCEEDED" for r in requests)
                measurement.update(
                    {
                        "success": result.completed
                        and completed == len(run.scenario.requests)
                        and len(requests) == len(run.scenario.requests),
                        "simulated_s": result.now_s,
                        "request_count": len(requests),
                        "completed_requests": completed,
                        "stage_count": len(stages),
                        "completed_stages": sum(s.status == "SUCCEEDED" for s in stages),
                        "advance_calls": 1,
                        "batch_advance_wall_s": advance_wall_s,
                        "event_count": result.metrics.events,
                        "retained_event_count": len(result.events),
                        "metrics": result.metrics.model_dump(mode="json"),
                    }
                )
    except Exception as exc:
        for measurement in measurements.values():
            if not measurement["success"]:
                measurement["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for pid in pids:
            sampler.unregister(pid)
    return list(measurements.values())


def measure_cell(
    node_count: int, workload: str, workers: int, repeats: int, ipc_samples: int, trace: bool
) -> CellMeasurement:
    scenario = benchmark_scenario(node_count, workload)
    runs = [
        RunSpec(
            scenario=scenario,
            run_id=f"{workload}-{node_count}-{workers}-{i}",
            seed=0,
            trace=trace,
            policy=PolicySpec(name="fifo"),
        )
        for i in range(workers * repeats)
    ]
    sampler = RssSampler()
    sampler.thread.start()
    began = perf_counter()
    measurements = []
    for offset in range(0, len(runs), workers):
        measurements.extend(measure_wave(runs[offset : offset + workers], sampler, ipc_samples))
    elapsed = perf_counter() - began
    sampler.stop.set()
    sampler.thread.join()
    successes = sum(m["success"] for m in measurements)
    completed = sum(m.get("completed_requests", 0) for m in measurements)
    startups = [m["startup_s"] for m in measurements if "startup_s" in m]
    ipc = sorted(t for m in measurements for t in m.get("inspect_rpc_s", []))
    return CellMeasurement.model_validate(
        {
            "nodes": node_count,
            "workload": workload,
            "workers": workers,
            "requested_runs": len(runs),
            "successful_runs": successes,
            "failed_runs": len(runs) - successes,
            "requests_per_run": len(scenario.requests),
            "completed_requests": completed,
            "wall_s": elapsed,
            "successful_runs_per_wall_s": successes / elapsed,
            "completed_requests_per_wall_s": completed / elapsed,
            "startup_median_s": statistics.median(startups) if startups else None,
            "inspect_rpc_median_s": statistics.median(ipc) if ipc else None,
            "inspect_rpc_p95_s": ipc[max(0, (95 * len(ipc) + 99) // 100 - 1)] if ipc else None,
            **sampler.summary(),
            "runs": measurements,
        }
    )


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--nodes", nargs="+", type=int, choices=(32, 128))
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("compute", "network", "mixed"),
    )
    parser.add_argument("--workers", nargs="+", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--repeats", type=positive, help="runs per worker per matrix cell")
    parser.add_argument("--ipc-samples", type=positive)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, help="compare against a compatible JSON report")
    parser.add_argument("--allow-regression", help="explicit explanation for accepted regressions")
    args = BenchmarkSettings(**vars(parser.parse_args()))
    if args.allow_regression and args.baseline is None:
        parser.error("--allow-regression requires --baseline")
    if args.baseline and args.baseline.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --baseline")
    baseline = (
        BenchmarkReport.model_validate_json(args.baseline.read_text()) if args.baseline else None
    )
    report = BenchmarkReport.model_validate(
        {
            "schema_version": 1,
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
                "cpu_count": os.cpu_count(),
                "simgrid": version("simgrid"),
                "trace": args.trace,
            },
            "methodology": {
                "concurrency": "BatchRunner waves; sequential submit/startup, "
                "concurrent advance_all",
                "startup": "submit() and session() through ready; "
                "includes spawn and engine/platform setup",
                "ipc": "inspect() round trip, including snapshot construction and serialization",
                "rss": "sampled parent + registered ready SDK workers; "
                "excludes startup before ready, "
                "resource tracker, ps process, and descendants; "
                "summed RSS double-counts shared pages",
                "throughput": "completed requests / full cell wall time "
                "including startup, RPC probes, "
                "execution, log retrieval and teardown; scenario construction excluded",
                "warmup": "none; same parent across all cells; dependency/import caches may warm",
            },
            "cells": [],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for node_count in args.nodes:
        for workload in args.workloads:
            for workers in args.workers:
                cell = measure_cell(
                    node_count, workload, workers, args.repeats, args.ipc_samples, args.trace
                )
                report = report.model_copy(update={"cells": (*report.cells, cell)})
                args.output.write_text(report.model_dump_json(indent=2) + "\n")
                print(
                    f"nodes={node_count} workload={workload} workers={workers} "
                    f"ok={cell.successful_runs}/{cell.requested_runs} "
                    f"requests/s={cell.completed_requests_per_wall_s:.2f}",
                    flush=True,
                )
    if baseline is not None:
        comparison = compare_reports(report, baseline, str(args.baseline), args.allow_regression)
        report = report.model_copy(update={"comparison": comparison})
        args.output.write_text(report.model_dump_json(indent=2) + "\n")
        print(comparison.model_dump_json(indent=2))
        if comparison.status in {"invalid", "regressed"}:
            raise SystemExit(1)
    if any(cell.failed_runs for cell in report.cells):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
