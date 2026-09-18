# Local baseline observation

The full matrix was measured on 2026-09-18 with:

```sh
uv run python -m benchmarks.run --repeats 1 --ipc-samples 2 --output benchmark-results/baseline.json
```

All 24 cells completed: 90 independent runs, 7,110 completed requests, no failed
runs, and no RSS sampling errors. This is one wave per cell on the recorded
macOS ARM64 host, Python 3.11.14, eight reported CPUs, SimGrid 4.1, tracing off.
The raw local report is `benchmark-results/baseline.json` and is Git-ignored.

| Measured cost across cells | 32 nodes | 128 nodes | Total |
| --- | ---: | ---: | ---: |
| Full cell wall time | 6.596 s | 7.782 s | 14.378 s |
| Sum of sequential worker startup timings | 5.244 s | 5.363 s | 10.607 s |
| Advance barriers, each wave counted once | 1.222 s | 2.054 s | 3.276 s |

Startup accounts for about 74% of the summed cell wall time in this workload.
The median worker startup was 115 ms; the median inspect RPC round trip was
2.36 ms. Peak sampled parent-plus-worker RSS across cells was 492,224,512 bytes
(about 469 MiB). Worker registration begins after readiness, so this does not
capture pre-ready worker memory peaks.

Startup is the largest measured component here. It includes imports, process
spawn, platform creation, and the ready handshake, and occurs sequentially in
this harness. Advance barriers include simulation, serialization, result
retrieval, and worker teardown: the 3.276 s is not isolated engine compute time.
The remaining wall time includes probes and driver overhead. Simulated FLOPs
are work units advanced by the engine, not actual host arithmetic operations.

These results identify where this small-workload run spent time. They do not
establish worker scaling, host CPU saturation, or performance on larger graphs.
The separate barrier timing cannot attribute cost between Python orchestration,
SimGrid, serialization, and teardown without additional profiling. The
[benchmark methodology](../benchmarks/README.md) defines the exact measurement
scope and regression-comparison thresholds.

The subsequent Pydantic report/CLI comparison smoke run flagged approximately
25% higher sampled RSS for the 32-node compute, one-worker cell, with essentially
unchanged throughput. The CLI correctly exited nonzero. A separate diagnostic
invocation verified that a nonempty `--allow-regression` explanation preserves
the flag while returning success. This was an override-path check, not a finding
that the RSS change is acceptable; its cause remains uninvestigated. The original
full baseline measurements were preserved.

## Final implementation rerun

After correctness fixes and full `RunSpec` manifests, a rerun identified an
unnecessary second result RPC in `BatchRunner`: terminal `AdvanceResult` already
contains that result. The runner now caches this same immutable object before
releasing its worker, avoiding duplicate serialization and retained result copies.
A transport regression test verifies the result object is reused.

The final comparison report is `benchmark-results/final-deduplicated.json`:
24 cells, 90/90 runs, 7,110 completions, 15.430 s summed cell wall time,
11.237 s summed startup, and 3.650 s summed advance barriers. No cell exceeds
the 15% throughput-decline threshold (worst change: -14.08%). Peak sampled
parent-plus-worker RSS was 538,312,704 bytes.

Seven cells still exceed the 20% RSS-growth threshold, so the comparison command
correctly exits nonzero; no override was applied. The largest relative change,
128-node mixed/one-worker, is +35.22%: sampled parent peak grew from 67,485,696
to 108,363,776 bytes, while worker peak grew from 53,198,848 to 54,820,864 bytes.
Thus the observed increase is primarily in the parent, not per-engine native
memory. Full manifests and richer retained Pydantic results expand the parent's
allocation workload; Python allocator retention across matrix cells is a plausible
additional contributor, not a verified leak diagnosis. Isolated-cell repetitions
and allocation profiling are required to attribute the remaining increase.

This is a functional V1 baseline, not a claim that all performance gates passed.
The earlier baseline and both before/after optimization reports remain available
locally. Do not replace the baseline simply to silence these memory warnings.
