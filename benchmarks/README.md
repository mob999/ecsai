# Benchmark methodology

Run from the repository root after `uv sync --all-packages`:

```sh
uv run python -m benchmarks.run --output benchmarks/results.json
```

The default matrix covers **32/128 nodes**, **compute/network/mixed**, and
**1/2/4/8 concurrent workers**, with two runs per worker per cell. Every worker
owns an independent simulated topology. Concurrency does not partition one
simulation. Each repeat submits one full wave to the public `BatchRunner`:
`submit()` boots workers sequentially, then `advance_all()` advances them
concurrently using the SDK's pipe readiness loop. Only RSS sampling uses a thread.
This exercises the actual batch API; it does not measure parallel startup or
the runner's queue-promotion path.

For a short environment check:

```sh
uv run python -m benchmarks.run --nodes 32 --workloads compute --workers 1 --repeats 1
```

Each scenario has one source node and one request per remaining node (31 or
127 requests). Each request has one stage constrained to its own edge node.
Compute work is 1 GFLOP on a 1 GFLOP/s, one-core node. Network work fetches a
1 MB shared object through a shared 100 MB/s link with 1 ms latency. Mixed work
does both. Network-only work has zero FLOPs. All stages reserve 1 MB working
memory. Network contention is deliberately concentrated on a shared link.
There is no real payload allocation or network traffic.

The JSON report records:

- Requested, successful, and failed runs; request and stage completion counts;
  advance calls; engine and retained event counts; run metrics; and each run's
  simulated time. The runner closes terminal sessions, so this harness does not
  retrieve command logs after completion.
- Full cell wall time and successful runs/completed requests per wall second.
  These denominators include spawning, probes, simulation, log retrieval, and
  teardown, but exclude scenario construction. Partial failed runs may contribute
  completed requests; inspect failure counts before comparing throughput.
- Startup duration across `submit()` and obtaining its ready session. This includes process
  spawn, imports, platform setup, and engine initialization, not just IPC.
- Individual `inspect()` RPC round trips before simulation advances, with median
  and nearest-rank p95. These include snapshot construction, serialization,
  transport, deserialization, and scheduling. They are **not pure IPC latency**.
- `batch_advance_wall_s` measures the whole wave's advance barrier, result
  retrieval, and terminal worker teardown. It is repeated in each run record
  for that wave, not an independent per-run duration; do not sum those values.
- Sampled parent RSS, summed worker RSS, and parent-plus-workers RSS, in bytes.
  Combined peak is the maximum of same-sample sums, not the sum of independent
  maxima. macOS/Linux `ps` reports KiB, which the harness converts to bytes.

RSS sampling uses a 20 ms wait plus the time to invoke `ps`. The report includes
how many samples contained workers and the maximum observed worker count; worker
peak RSS is null if no workers were observed. It can miss short
peaks. Workers become registered after the ready handshake, so startup worker
RSS before readiness is excluded. Python's resource tracker, the `ps` process,
and other descendants are excluded. Summed RSS counts shared pages more than
once; it is not unique physical memory. Sampling failures are recorded rather
than reported as zero. The parent includes imported modules, immutable scenario
data, accumulated results, and measurement overhead; it is not baseline-subtracted.

Tracing defaults off; use `--trace` to measure trace retention explicitly.
`event_count` comes from engine metrics; `retained_event_count` is the saved log
length and may be zero with tracing disabled. There
is no warmup, and later cells can benefit from operating-system caches. Keep raw
per-run data, host metadata, tracing settings, and failures with any published
comparison. The harness exits nonzero if any run fails and writes the report
after each completed matrix cell. Results are measurements of this workload,
not a claim about calibrated real-network performance.

## Baseline comparison

```sh
uv run python -m benchmarks.run --repeats 1 --ipc-samples 2 --output benchmark-results/baseline.json
uv run python -m benchmarks.run --repeats 1 --ipc-samples 2 --baseline benchmark-results/baseline.json --output benchmarks/comparison.json
```

The second invocation measures a new run and writes a `comparison` in its JSON
report. It requires matching platform, Python version, CPU count, SimGrid version,
and tracing mode. Each selected current cell must exist in the baseline, with
matching run/request counts and usable successful measurements. A subset of a
larger baseline is allowed. Failed runs, missing RSS evidence, sampling errors,
or incompatible environments invalidate the comparison and exit nonzero.

Each cell flags completed-request throughput **below 85%** of baseline or sampled
parent-plus-worker RSS **above 120%** of baseline. Exactly -15% and +20% pass.
Unexplained regressions exit nonzero. To explicitly accept measured regressions,
add `--allow-regression "reason for accepting this change"`. The report retains
the deltas, flags, and explanation with status `allowed`. This flag cannot waive
environment mismatches, failed runs, or missing evidence. The output path cannot
overwrite the baseline. Raw JSON outputs are ignored by Git.

Configuration uses Pydantic `BenchmarkSettings` (`BaseSettings`), and saved
environment, methodology, measurements, and comparisons are Pydantic models.
CLI values override `EDGE_BENCH_` environment settings; for example,
`EDGE_BENCH_REPEATS=1`. Tuple settings use JSON arrays when supplied through the
environment. These benchmark settings are separate from SDK `EDGE_SIM_` settings.

These thresholds are regression flags, not statistical significance tests.
Use consistent repeat/probe counts and host load, and investigate flagged cells
before accepting them. One wave per cell does not establish scaling behavior.
