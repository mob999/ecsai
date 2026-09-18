# SDK recipes

The examples are the executable reference for scenario construction. All model
constructors below are exported by `edge_sim_models`; the SDK exports `Session`,
`start`, `BatchRunner`, `SDKError`, `Settings`, and `validate`.

## Validation and units

`validate(scenario_or_dict)` returns a frozen `ScenarioSpec` or raises Pydantic's
`ValidationError`. Nodes use positive `speed_flops`, byte capacities, and an
integer core count. Links use positive `bandwidth_bytes_s` and nonnegative
`latency_s`. Stages may have zero FLOPs, which is useful for network-only work.

An artifact produced by a stage must name that stage in `producer_stage`, and
the stage must declare it in `outputs`. Every external workflow input must have
an `InputBinding(artifact_id=..., global_artifact_id=...)` on the request. The
global object and local artifact must have matching sizes. Routes are directed.

## External decisions

Create `RunSpec(..., external=True)`. After `advance()` returns a decision,
read `step.decision.view`, choose typed commands, and call
`apply(step.decision.decision_id, tuple_of_commands)`. Decision IDs are scoped to
the run's current revision; do not reuse one after applying its commands. An
empty command batch is not a wait operation: use `Defer(until_s=future_time)`.
`session.commands()` returns Pydantic command records; serialize records with
`model_dump()` or `model_dump_json()` rather than treating them as dictionaries.

The [preemption example](../examples/external_preemption.py) places background
work at time 0, defers to an urgent arrival at time 1, suspends background work,
places urgent work, and resumes the background work after the urgent stage
succeeds. It checks that the suspended stage keeps its working-memory reservation.
There is no migration in this example; both stages use the same node.

`advance(until_time=t)` may stop with `kind="time"`, but a pending external
decision still needs a response. Always branch on `kind`, not just the simulated
clock. `result()` is also available before termination; inspect `completed`,
`end_reason`, and request statuses before treating it as a successful result.

## Independent batches

```python
from edge_sim import BatchRunner
from edge_sim_models import PolicySpec, RunSpec
from examples.scenarios import content_delivery


def main():
    scenario = content_delivery()
    with BatchRunner(workers=2) as runner:
        for index in range(4):
            runner.submit(
                RunSpec(
                    scenario=scenario,
                    run_id=f"run-{index}",
                    seed=index,
                    policy=PolicySpec(name="fifo"),
                )
            )
        results = runner.run()
        print({run_id: result.metrics.completed for run_id, result in results.items()})


if __name__ == "__main__":
    main()
```

`run()` drains all submitted runs using built-in policies. For explicit stepping,
`advance_all()` is a barrier over the currently active runs, not the whole queued
batch. `submit_advance(run_id)` and `recv_ready(timeout=...)` expose readiness
without a barrier. Returned `SDKError` values must be handled per run.
For an external decision, use `runner.session(run_id).apply(...)` after receiving
the decision. A session allows only one outstanding RPC; collect its response
before inspecting or applying. Finished results are cached for
`runner.result(run_id)` after the worker closes.

## Host settings and errors

`Settings` is a frozen Pydantic `BaseSettings` model. Its environment prefix is
`EDGE_SIM_`: `EDGE_SIM_WORKERS`, `EDGE_SIM_TIMEOUT_S`, and `EDGE_SIM_OUTPUT_DIR`.
These are host execution settings, separate from simulated cores, capacities,
and time. For explicit programmatic configuration, pass settings values to the
session or batch constructor:

```python
from edge_sim import BatchRunner, Settings

settings = Settings()
runner = BatchRunner(workers=settings.workers, timeout_s=settings.timeout_s)
```

Use the runner as a context manager when submitting work. `SDKError.code` and
`SDKError.message` identify transport and backend failures. A transport failure
closes the session; commands are not automatically replayed. Close a session and
start a fresh run to reset. Serialized views and event logs are not checkpoints.
