# Edge Simulation

A Python workspace for process-isolated edge workflow simulation with SimGrid.
Scenarios describe compute nodes, directed network routes, artifact replicas,
request arrivals, and workflow DAGs. Policies place and preempt stage execution;
SimGrid advances compute and network activities.

## Install

Python 3.11 or 3.12 and `uv` are required. Install `uv` using your package manager
or the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Run installation commands from the repository root. The backend pins SimGrid 4.1.
Bootstrap its native wheel before syncing the workspace: upstream's build can
select a different Python ABI, and pybind11 3 is incompatible with its bindings.

macOS:

```sh
brew install uv boost cmake
uv venv --python 3.11 .venv
.venv/bin/python scripts/build_simgrid.py
uv sync --locked --all-packages
```

Linux (Debian/Ubuntu, with `uv` already installed):

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-dev
uv venv --python 3.11 .venv
.venv/bin/python scripts/build_simgrid.py
uv sync --locked --all-packages
```

The [bootstrap script](scripts/build_simgrid.py) verifies the source hash from
`uv.lock`, builds through `uv build` with pybind11 2.13.6 and matching Python
development files, and installs the resulting wheel. It configures native library
lookup beside the extension (`@loader_path` on macOS, `$ORIGIN` on Linux), then
checks import and an asynchronous compute/transfer simulation with no library-path
environment overrides. Wheels are saved in `dist/simgrid/` for the current Python,
OS, and architecture. CMake, Boost, and the C++ compiler must be installed separately.

Subsequent `uv sync --locked --all-packages` preserves the repaired installation.
After recreating `.venv` or explicitly reinstalling SimGrid from PyPI, rerun the
bootstrap before syncing. CI follows this sequence and checks that syncing leaves
the native extension unchanged. Verify an existing environment with:

```sh
uv run --locked python -I -c 'import simgrid; print(simgrid.simgrid_version, simgrid.__file__)'
```

Run all commands below from the repository root. Scripts that create sessions
must use `if __name__ == "__main__":` because workers use multiprocessing spawn.

## DEPPO / BenchMARL learning

The optional `edge-sim-learning` package provides periodic content-service simulation,
DEPPO-adapted / MAPPO-no-context training, drained evaluation and W&B logging.
See [the experiment guide](docs/deppo.md) for CPU smoke, Linux CUDA training,
three-seed validation and the paper-to-implementation differences.

## Examples

```sh
uv run python -m examples.content_delivery
uv run python -m examples.dag_fork_join
uv run python -m examples.external_preemption
uv run python -m examples.policy_plugin
```

- Content delivery binds two request-local inputs to the same global object,
  fetches it from the origin, and serves it on the edge.
- Fork/join runs independent DAG branches on different nodes and waits for both
  outputs before executing the join.
- External preemption uses `Place`, `Suspend`, `Resume`, and `Defer` to interrupt
  background work for an urgent arrival on a single-core node.

Each example checks that requests complete and prints its result. These are
simulation examples; artifact bytes describe sizes, not real media payloads.
The plugin example loads a trusted local Pydantic placement policy through
`PolicySpec.plugins`; see the [plugin contracts](docs/plugins.md).

## Session API

```python
from edge_sim import start
from edge_sim_models import RunSpec
from examples.scenarios import content_delivery


def main():
    run = RunSpec(scenario=content_delivery(), run_id="demo")
    with start(run) as session:
        step = session.advance()
        while step.kind != "finished":
            # This run uses a built-in policy, so no external decision is expected.
            if step.kind == "decision":
                raise RuntimeError("An external policy must answer this decision")
            step = session.advance()
        print(session.result().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

`advance()` returns a decision, a requested time boundary, or a finished result.
External policy code responds with
`session.apply(step.decision.decision_id, commands)`.
`inspect()`, `result()`, `events()`, and `commands()` expose snapshots and records.
Use a new session for a fresh run; there is no checkpoint or in-process reset API.

The immutable contracts live in `edge_sim_models`, the process facade in
`edge_sim`, and the engine implementation in `edge_sim_simgrid`. See the
[SDK recipes](docs/sdk.md) for external decisions, batch execution, and settings,
and the [architecture and limitations](docs/architecture.md) for ownership, data
separation, memory pinning, and raw network model semantics.

## Verification and benchmarks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m benchmarks.run --output benchmarks/results.json
```

The benchmark matrix covers 32/128 nodes, compute/network/mixed workloads, and
1/2/4/8 concurrent independent workers. It reports completion counts, throughput,
startup, inspect RPC round trips, and sampled parent-plus-worker RSS. Read the
[methodology](benchmarks/README.md) before interpreting the numbers: RSS is
sampled, RPC timings include serialization, and startup is part of throughput.
The [local baseline observation](docs/benchmark-baseline.md) separates measured
startup and advance costs. Use `--baseline path/to/report.json` to compare a new
run with matching environment metadata; threshold and override behavior are
documented in the benchmark methodology.
