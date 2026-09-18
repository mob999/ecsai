"""Small deterministic scenarios using only public model contracts."""

from edge_sim_models import (
    ArtifactSpec,
    InputBinding,
    LinkSpec,
    NodeSpec,
    RequestSpec,
    RouteSpec,
    ScenarioSpec,
    StageSpec,
    WorkflowArtifactSpec,
    WorkflowSpec,
)


def nodes(count: int) -> tuple[NodeSpec, ...]:
    return tuple(
        NodeSpec(
            id=f"n{i}",
            role="cloud" if i == 0 else "edge",
            cores=1,
            speed_flops=1e9,
            memory_bytes=256_000_000,
            storage_bytes=1_000_000_000,
        )
        for i in range(count)
    )


def content_delivery() -> ScenarioSpec:
    """Fetch a shared global object onto an edge node before serving it."""
    return ScenarioSpec(
        nodes=nodes(2),
        links=(LinkSpec(id="uplink", bandwidth_bytes_s=10_000_000, latency_s=0.01),),
        routes=(RouteSpec(src="n0", dst="n1", links=("uplink",)),),
        artifacts=(ArtifactSpec(id="video", size_bytes=10_000_000, locations=("n0",)),),
        workflows=(
            WorkflowSpec(
                id="delivery",
                artifacts=(WorkflowArtifactSpec(id="content", size_bytes=10_000_000),),
                stages=(
                    StageSpec(
                        id="serve",
                        flops=1e8,
                        memory_bytes=1_000_000,
                        inputs=("content",),
                        eligible_nodes=("n1",),
                    ),
                ),
            ),
        ),
        requests=tuple(
            RequestSpec(
                id=f"watch-{i}",
                workflow="delivery",
                receiver="n1",
                arrival_s=i * 2,
                input_bindings=(InputBinding(artifact_id="content", global_artifact_id="video"),),
            )
            for i in range(2)
        ),
    )


def fork_join() -> ScenarioSpec:
    """A root feeds two branches on separate nodes, then joins their outputs."""
    return ScenarioSpec(
        nodes=nodes(3),
        links=(LinkSpec(id="fabric", bandwidth_bytes_s=100_000_000, latency_s=0.001),),
        routes=tuple(
            RouteSpec(src=f"n{i}", dst=f"n{j}", links=("fabric",))
            for i in range(3)
            for j in range(3)
            if i != j
        ),
        workflows=(
            WorkflowSpec(
                id="fork-join",
                artifacts=tuple(
                    WorkflowArtifactSpec(
                        id=artifact,
                        size_bytes=100_000,
                        producer_stage=producer,
                    )
                    for artifact, producer in (
                        ("seed", "root"),
                        ("left-out", "left"),
                        ("right-out", "right"),
                    )
                ),
                stages=(
                    StageSpec(id="root", flops=1e8, outputs=("seed",), eligible_nodes=("n0",)),
                    StageSpec(
                        id="left",
                        flops=1e9,
                        inputs=("seed",),
                        outputs=("left-out",),
                        depends_on=("root",),
                        eligible_nodes=("n1",),
                    ),
                    StageSpec(
                        id="right",
                        flops=2e9,
                        inputs=("seed",),
                        outputs=("right-out",),
                        depends_on=("root",),
                        eligible_nodes=("n2",),
                    ),
                    StageSpec(
                        id="join",
                        flops=1e8,
                        inputs=("left-out", "right-out"),
                        depends_on=("left", "right"),
                        eligible_nodes=("n0",),
                    ),
                ),
            ),
        ),
        requests=(RequestSpec(id="dag", workflow="fork-join", receiver="n0"),),
    )


def preemption() -> ScenarioSpec:
    return ScenarioSpec(
        nodes=nodes(1),
        workflows=tuple(
            WorkflowSpec(
                id=name,
                stages=(StageSpec(id="compute", flops=flops, memory_bytes=1_000_000),),
            )
            for name, flops in (("long", 10e9), ("urgent", 1e9))
        ),
        requests=(
            RequestSpec(id="background", workflow="long", receiver="n0"),
            RequestSpec(id="urgent", workflow="urgent", receiver="n0", arrival_s=1),
        ),
    )


def benchmark_scenario(node_count: int, workload: str) -> ScenarioSpec:
    """One request per edge, all transfers share one bottleneck link."""
    if node_count < 2 or workload not in {"compute", "network", "mixed"}:
        raise ValueError("expected >=2 nodes and compute/network/mixed workload")
    network = workload != "compute"
    compute = workload != "network"
    size = 1_000_000
    return ScenarioSpec(
        nodes=nodes(node_count),
        links=(LinkSpec(id="shared", bandwidth_bytes_s=100_000_000, latency_s=0.001),),
        routes=tuple(
            RouteSpec(src="n0", dst=f"n{i}", links=("shared",)) for i in range(1, node_count)
        ),
        artifacts=(ArtifactSpec(id="source", size_bytes=size, locations=("n0",)),)
        if network
        else (),
        workflows=tuple(
            WorkflowSpec(
                id=f"work-{i}",
                artifacts=(WorkflowArtifactSpec(id="input", size_bytes=size),) if network else (),
                stages=(
                    StageSpec(
                        id="work",
                        flops=1e9 if compute else 0,
                        memory_bytes=1_000_000,
                        inputs=("input",) if network else (),
                        eligible_nodes=(f"n{i}",),
                    ),
                ),
            )
            for i in range(1, node_count)
        ),
        requests=tuple(
            RequestSpec(
                id=f"request-{i}",
                workflow=f"work-{i}",
                receiver=f"n{i}",
                input_bindings=(InputBinding(artifact_id="input", global_artifact_id="source"),)
                if network
                else (),
            )
            for i in range(1, node_count)
        ),
    )
