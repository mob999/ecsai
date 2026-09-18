# Policy plugins

`edge_sim_models.policies` defines Python `Protocol` interfaces for six policy
roles. Protocols describe behavior; inputs, commands, configuration, and state
views use Pydantic models. `PluginSpec` and `PolicySpec` are frozen Pydantic DTOs.
Use Pydantic for plugin data models and `BaseSettings` for environment-backed settings.

Configure a trusted importable, zero-argument factory:

```python
from edge_sim_models import PluginSpec, PolicySpec

policy = PolicySpec(
    name="fifo",
    plugins=(
        PluginSpec(
            role="placement",
            factory="examples.policy_plugin:make_policy",
        ),
    ),
)
```

Pass this policy to `RunSpec`. Each spawned worker imports the module, calls the
factory once, and keeps the returned object for that run. The factory must be
accessible by its module and attribute name in the worker environment. Current
`PluginSpec` fields are `role` and `factory`; there is no factory-kwargs or
configuration-payload field. A factory can construct its own Pydantic settings.

Factory strings are trusted code configuration, not sandboxed scenario data:
module imports and factories execute Python with worker process permissions.
Do not accept arbitrary factory paths from untrusted input. Only one plugin per
role is allowed; roles without overrides retain the backend defaults.

| Role / protocol | Method | Return contract |
| --- | --- | --- |
| `admission` / `AdmissionPolicy` | `admit(request: RequestSpec, view: StateView)` | `bool`; false rejects the arriving request |
| `placement` / `PlacementPolicy` | `decide(decision: DecisionRequest)` | Tuple of typed decision commands using current candidates |
| `replica` / `ReplicaPolicy` | `select(artifact_id: str, destination: str, candidates: tuple[str, ...])` | One source node from the offered candidates |
| `scheduling` / `SchedulingPolicy` | `order(stages: tuple[StageState, ...])` | A permutation of offered `request_id/stage_id` keys |
| `preemption` / `PreemptionPolicy` | `decide(view: StateView)` | Tuple of `Suspend`/`Resume` commands; empty means no action |
| `cache` / `CachePolicy` | `order(node_id: str, evictable_ids: tuple[str, ...])` | A permutation of supplied, already LRU-sorted unpinned IDs |

Commands pass through backend validation. Scheduling and cache policies may
reorder candidates but may not add or remove them. Replica selection must return
an available, reachable source from its candidates. Cache policy does not
override pinning or capacity. Preemption runs at domain event boundaries, not
from a wall-clock polling loop.

The placement hook is part of built-in-policy execution. For parent-driven
interactive placement, use `RunSpec(external=True)` and the session decision
loop instead. Avoid combining the two placement control paths. Plugins receive
domain views, not SimGrid engine handles.

The [runnable Pydantic plugin example](../examples/policy_plugin.py) selects the
first eligible node for one stage per decision:

```sh
uv run python -m examples.policy_plugin
```

This demonstrates the extension boundary, not an optimized placement strategy.
