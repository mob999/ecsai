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
