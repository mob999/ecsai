# Architecture and implementation plan

This document records the agreed design. Runtime claims must be checked against
the implementation and integration tests; a contract alone does not establish
that a backend supports every valid scenario.

## Package boundaries

| Package | Responsibility | Must not own |
| --- | --- | --- |
| `edge_sim_models` | Frozen Pydantic input, command, state and result DTOs; reference and DAG validation | SimGrid objects or simulation progress |
| `edge_sim` | Session lifecycle, spawned worker transport, policy-facing API, independent run orchestration | A second CPU or network simulator |
| `edge_sim_simgrid` | Platform conversion, one engine per worker, activities, resource accounting, trace production | Mutable objects shared with caller code |

The parent validates a `ScenarioSpec`, wraps it in a `RunSpec`, then calls
`start(run)`. A spawned child constructs the SimGrid engine and reports readiness.
The parent and worker exchange serialized DTOs over a multiprocessing pipe.
Only the worker imports and initializes SimGrid. Use a guarded `main` entrypoint
in scripts because Python's spawn method imports the launching module.

## Data separation

There are four separate kinds of data:

1. Scenario definitions describe nodes, directed routes, global artifacts,
   workflows, and request arrivals. They are immutable value objects.
2. Backend state holds activity handles, reservation ledgers, replicas, transfer
   waiters, and lifecycle state. This state is private to one worker.
3. State views and results are immutable snapshots sent to the parent. Editing a
   copy cannot alter a running engine.
4. Events and commands record observations and decisions. A trace is an audit
   record, not a serialized engine checkpoint.

All data models use Pydantic; environment-backed settings use Pydantic Settings
`BaseSettings`. Tuple collections prevent mutable nested dictionaries from leaking through the
domain contract. Global input artifacts are shared objects with initial replica
locations. Workflow artifacts have local IDs; a request binds external workflow
inputs to global IDs through `InputBinding`. Produced artifacts belong to a
request, even when two requests use the same workflow and local artifact IDs.

This separation is logical and process-level. It does not provide zero-copy
transport, distributed storage, payload confidentiality, or fault-tolerant
recovery. Artifacts model byte counts and locations, not actual file contents.
Large snapshots and full traces incur serialization and parent memory costs.

## Execution and decisions

Stages specify FLOPs, working memory, input/output artifacts, explicit predecessor
stages, and optional eligible nodes. Data dependencies also form DAG edges.
Validation checks IDs, references, input bindings, and dependency cycles before
engine execution. Times are seconds, storage and memory are bytes, bandwidth is
bytes per second, and node speed is FLOPs per second.

`Session.advance(until_time=None)` returns `AdvanceResult` with one of:

- `kind="decision"`: an external policy must consume `decision.view` and call
  `apply(decision.decision_id, commands)` using that decision's ID.
- `kind="time"`: the requested simulated-time boundary was reached; this does
  not imply that all requests have completed.
- `kind="finished"`: `result` contains the terminal run snapshot. Inspect request
  statuses and `completed`, especially for bounded runs.

`Place` assigns a ready stage to a node; `Suspend` and `Resume` act on an existing
execution; `Reject` terminates a request; `Defer` chooses a future policy wakeup.
The external policy runs while the controller is waiting on the pipe, so policy
wall time does not itself advance simulated time. `inspect()`, `result()`,
`events()`, and `commands()` retrieve snapshots or logs. They are not stepping
operations. Exact command eligibility and accounting are backend constraints.

Built-in policy names are `fifo`, `edf`, and `round_robin`. External policies use
`RunSpec(external=True)`. Built-in policies use, for example, `PolicySpec(name="fifo")`.
Public states use uppercase names: `RUNNING`, `SUSPENDED`, and successful terminal
`SUCCEEDED`. Run metrics include `arrived`, `completed`, `rejected`, `failed`,
`timed_out`, and `unfinished`, plus `Metric(name, value)` tuples for CPU/link
utilization. Results do not define a run reward.
No example should assume an unimplemented callback,
checkpoint, stage-migration, or direct engine-access API.

Trusted local policy factories are configured using `PolicySpec.plugins` and
typed `PluginSpec(role, factory)` entries. The worker loads them separately from
the parent-driven decision loop. See [plugin contracts](plugins.md) for admission,
placement, replica, scheduling, preemption, and cache interfaces.

## Resource and network semantics

SimGrid owns computational and communication progress and contention. Python
ledgers enforce admission capacity; they must not duplicate progress with manual
completion timers. Memory reservations and storage reservations are distinct.
Suspension is compute preemption on the assigned node, not migration or a saved
checkpoint. The intended memory rule is that suspended work retains its working
memory reservation; scheduling must account for that pinned capacity.

Artifact pinning is also distinct from working memory: replicas that are needed
by in-flight work cannot be treated as freely evictable cache entries. Storage
pressure, cacheability, pinning, and transfer coalescing need explicit accounting.
Do not interpret every available replica as independently evictable.
Initial source replicas remain pinned. Other replicas preserve local active inputs
and the last reachable source for future consumers; redundant remote copies can
be reclaimed. Data has no disk I/O latency in V1.

The worker configures SimGrid `network/model:raw` and `cpu/model:Cas01`. Raw network
timings are model-specific and should not be presented as a calibrated TCP or
packet-level prediction. Routes are directed; shared link IDs express contention.
Explicit reverse routes are needed for reverse traffic. Route completeness,
replica reachability, storage capacity, and placement feasibility remain relevant
even for a structurally valid DAG.

There is one narrow backend workaround: SimGrid 4.1 raw `sendto_async` activities
with zero bytes can remain incomplete indefinitely. For these transfers only,
the runtime tracks a simulated-time timer equal to the sum of the directed
route's link latencies. A zero-latency delivery completes at the current time;
positive latency participates in the controller's normal SimGrid wait/sleep
boundaries. These transfers consume no bandwidth, but retain normal replica
pinning, waiter coalescing, cancellation, and transfer events. Completion at a
deadline wins the tie. Positive-byte transfers and computational progress remain
owned by SimGrid; no byte-progress estimator or replacement network model is used.

## Isolation, reset, and limitations

One session owns one spawned process and one engine. Closing the context tears
down that worker. Reset means closing the old session and calling `start()` with
a new run, paying startup costs again. There is no in-process reset, checkpoint,
resume-from-snapshot, or process migration contract. A seed is part of the run
record; it does not promise identical wall timings or cross-version numerical
identity.

Exported manifests retain the complete validated `RunSpec`, including overheads,
termination conditions, seeds, and policy version labels. Plugin versions are
caller-supplied metadata, not automatic source fingerprints. Archive plugin code
with experiment results when reproducibility across checkouts matters.

RPCs are synchronous and a session permits one outstanding operation. Transport
timeouts and dead workers become SDK errors; failed operations are not silently
retried. Context-manager cleanup is required even when policy code raises. Run
concurrency means multiple independent engines in separate workers, not multiple
workers cooperating on a single simulated topology.

## Delivery and verification plan

1. Freeze and inspect the model constructors and session boundary.
2. Implement content delivery, fork/join, and explicit preemption examples using
   only public DTOs and session methods.
3. Run the examples against the real backend and check terminal request states,
   dependency ordering, and the recorded preemption commands.
4. Benchmark 32/128-node compute, network, and mixed workloads with 1/2/4/8
   concurrent independent sessions. Keep scenario construction out of measured
   startup and identify all wall-time denominators.
5. Report actual completions, startup and inspect-RPC timings, and sampled parent
   plus worker RSS. Record failures rather than substituting estimates. The
   harness uses public `BatchRunner` waves: sequential startup and concurrent
   `advance_all()` barriers, with the distinction recorded in its methodology.

Benchmark results describe the measured host, dependency versions, topology,
workload, and tracing configuration. They are not hardware-independent capacity
claims. See [benchmark methodology](../benchmarks/README.md).
