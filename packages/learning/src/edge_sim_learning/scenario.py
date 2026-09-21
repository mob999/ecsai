"""Versioned synthetic workloads; policy-independent random streams."""

from typing import Literal

import numpy as np
from edge_sim_models import (
    ArtifactSpec,
    CacheNodeSpec,
    ContentRequest,
    ContentServiceSpec,
    LinkSpec,
    NodeSpec,
    RouteSpec,
    RunSpec,
    ScenarioSpec,
    SchedulerSpec,
    TransferPoolSpec,
)
from pydantic import BaseModel, ConfigDict, Field


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    scheduler_release: Literal["continuous", "window"] = "continuous"
    bandwidth_mode: Literal["independent", "shared"] = "shared"
    capacity_profile: Literal["calibrated", "paper-audit"] = "calibrated"
    delivery_load: float = Field(default=0.75, gt=0)
    topology_seed: int = 1729
    coalesce_backhaul: bool = False
    reward_mode: Literal["business", "paper", "logical"] = "business"
    reward_scale: float = Field(default=1.0, gt=0)
    clusters: int = Field(default=3, ge=2)
    caches: int = Field(default=10, ge=2)
    request_rate: float = Field(default=300, gt=0)
    episode_loads: tuple[float, ...] = ()
    catalog_size: int = Field(default=1000, ge=1)
    zipf_alpha: float = Field(default=1, gt=0)
    min_size: int = Field(default=500_000, ge=1)
    max_size: int = Field(default=7_500_000, ge=1)
    deadline_s: float = Field(default=1, gt=0)
    period_s: float = Field(default=0.1, gt=0)
    cycles: int = Field(default=128, ge=1)
    bandwidth_bytes_s: float = Field(default=125_000_000, gt=0)
    backhaul_bandwidth_bytes_s: float | None = Field(default=None, gt=0)
    delivery_bandwidth_bytes_s: float | None = Field(default=None, gt=0)
    backhaul_concurrency: int | None = Field(default=None, ge=1)
    delivery_concurrency: int | None = Field(default=None, ge=1)
    backhaul_waiting: int | None = Field(default=None, ge=0)
    delivery_waiting: int | None = Field(default=None, ge=0)
    cache_bytes: int = Field(default=100_000_000, ge=1)
    backhaul_latency_s: float = Field(default=0.02, ge=0)
    delivery_latency_s: float = Field(default=0.005, ge=0)
    queue_capacity: int = Field(default=50, ge=0)
    transfer_concurrency: int = Field(default=1, ge=1)
    scheduler_capacity: int = Field(default=100, ge=0)
    scheduler_service_s: float = Field(default=0.001, gt=0)
    max_retries: int = Field(default=0, ge=0)
    retry_delay_s: float = Field(default=0.1, ge=0)

    def load_phases(self):
        loads = self.episode_loads or (self.delivery_load,)
        if len(loads) > self.cycles or any(x <= 0 for x in loads):
            raise ValueError("episode loads must be positive with at least one cycle per phase")
        return [
            (
                i * self.cycles // len(loads) * self.period_s,
                (i + 1) * self.cycles // len(loads) * self.period_s,
                load,
                self.request_rate * load / self.delivery_load,
            )
            for i, load in enumerate(loads)
        ]

    def expected_arrivals(self, start_s, end_s):
        return sum(
            max(0, min(end_s, end) - max(start_s, start)) * rate
            for start, end, _, rate in self.load_phases()
        )

    @classmethod
    def profile(cls, name):
        if name == "smoke":
            return cls(
                clusters=2,
                caches=2,
                catalog_size=16,
                request_rate=30,
                min_size=50_000,
                max_size=500_000,
                cycles=8,
            )
        dimensions = {"small": (3, 10, 300), "medium": (5, 20, 1000), "large": (7, 30, 2000)}
        clusters, caches, rate = dimensions[name]
        return cls(clusters=clusters, caches=caches, request_rate=rate)

    @property
    def horizon_s(self):
        return self.cycles * self.period_s


def build_run(config: ScenarioConfig, seed: int, run_id="content", trace=False):
    if config.caches < config.clusters or config.max_size < config.min_size:
        raise ValueError("each cluster needs a cache and size bounds must be ordered")
    _, catalog_seed, arrivals_seed = np.random.SeedSequence(seed).spawn(3)
    topology_seed = config.topology_seed
    topo, catalog, arrivals = (
        np.random.default_rng(s) for s in (topology_seed, catalog_seed, arrivals_seed)
    )
    sizes = catalog.integers(config.min_size, config.max_size + 1, config.catalog_size)
    artifacts = tuple(
        ArtifactSpec(id=f"object-{i}", size_bytes=int(size), locations=("origin",))
        for i, size in enumerate(sizes)
    )
    nodes = [
        NodeSpec(
            id="origin",
            role="cloud",
            speed_flops=1e12,
            memory_bytes=0,
            storage_bytes=int(sizes.sum()),
        )
    ]
    schedulers = tuple(
        SchedulerSpec(
            id=f"cluster-{i}",
            max_waiting=config.scheduler_capacity,
            service_s=config.scheduler_service_s,
        )
        for i in range(config.clusters)
    )
    for i in range(config.clusters):
        nodes.append(
            NodeSpec(id=f"users-{i}", role="client", speed_flops=1, memory_bytes=0, storage_bytes=0)
        )
    caches, links, routes = [], [], []
    ownership = topo.permutation(config.caches) % config.clusters
    capacities = topo.uniform(1, 3, config.caches)
    if config.capacity_profile == "paper-audit":
        capacities *= 12_000_000 / 8  # Mbps -> bytes/s, not MB/s
    else:
        # Use the distribution expectation, independent of evaluation catalogue draws.
        capacities *= (
            config.request_rate
            * (config.min_size + config.max_size)
            / 2
            / config.delivery_load
            / capacities.sum()
        )
    pool = TransferPoolSpec(
        max_active=config.transfer_concurrency, max_waiting=config.queue_capacity
    )
    for i in range(config.caches):
        node, backhaul, delivery = f"cache-{i}", f"backhaul-{i}", f"delivery-{i}"
        nodes.append(
            NodeSpec(id=node, speed_flops=1e9, memory_bytes=0, storage_bytes=config.cache_bytes)
        )
        links.extend(
            (
                LinkSpec(
                    id=backhaul,
                    bandwidth_bytes_s=float(capacities[i] / 2)
                    if config.bandwidth_mode == "shared"
                    else config.backhaul_bandwidth_bytes_s or config.bandwidth_bytes_s,
                    latency_s=config.backhaul_latency_s,
                ),
                LinkSpec(
                    id=delivery,
                    bandwidth_bytes_s=float(capacities[i] / 2)
                    if config.bandwidth_mode == "shared"
                    else config.delivery_bandwidth_bytes_s or config.bandwidth_bytes_s,
                    latency_s=config.delivery_latency_s,
                ),
            )
        )
        routes.append(RouteSpec(src="origin", dst=node, links=(backhaul,)))
        routes.extend(
            RouteSpec(src=node, dst=f"users-{j}", links=(delivery,)) for j in range(config.clusters)
        )
        caches.append(
            CacheNodeSpec(
                node_id=node,
                cluster_id=f"cluster-{ownership[i]}",
                total_bandwidth_bytes_s=float(capacities[i])
                if config.bandwidth_mode == "shared"
                else None,
                backhaul_link=backhaul,
                delivery_link=delivery,
                backhaul=TransferPoolSpec(
                    max_active=config.backhaul_concurrency or pool.max_active,
                    max_waiting=pool.max_waiting
                    if config.backhaul_waiting is None
                    else config.backhaul_waiting,
                ),
                delivery=TransferPoolSpec(
                    max_active=config.delivery_concurrency or pool.max_active,
                    max_waiting=pool.max_waiting
                    if config.delivery_waiting is None
                    else config.delivery_waiting,
                ),
            )
        )
    popularity = np.arange(1, config.catalog_size + 1, dtype=float) ** -config.zipf_alpha
    popularity /= popularity.sum()
    requests = []
    for start, end, _, rate in config.load_phases():
        now = start
        while True:
            now += float(arrivals.exponential(1 / rate))
            if now >= end:
                break
            cluster = int(arrivals.integers(config.clusters))
            aid = int(arrivals.choice(config.catalog_size, p=popularity))
            requests.append(
                ContentRequest(
                    id=f"r-{len(requests)}",
                    artifact_id=f"object-{aid}",
                    cluster_id=f"cluster-{cluster}",
                    receiver=f"users-{cluster}",
                    arrival_s=now,
                    deadline_s=now + config.deadline_s,
                )
            )
    run = RunSpec(
        run_id=run_id,
        seed=seed,
        trace=trace,
        control_mode="window",
        scenario=ScenarioSpec(
            nodes=tuple(nodes), links=tuple(links), routes=tuple(routes), artifacts=artifacts
        ),
        content=ContentServiceSpec(
            origin="origin",
            max_retries=config.max_retries,
            retry_delay_s=config.retry_delay_s,
            scheduler_release=config.scheduler_release,
            bandwidth_mode=config.bandwidth_mode,
            coalesce_backhaul=config.coalesce_backhaul,
            schedulers=schedulers,
            caches=tuple(caches),
            requests=tuple(requests),
            size_scale_bytes=config.max_size,
            deadline_scale_s=config.deadline_s,
        ),
    )
    effective_rate = (
        config.expected_arrivals(0, config.horizon_s) / config.horizon_s
        if config.episode_loads
        else config.request_rate
    )
    metadata = {
        "expected_delivery_load": float(
            effective_rate
            * (sizes * popularity).sum()
            / (
                capacities.sum()
                if config.bandwidth_mode == "shared"
                else config.caches * (config.delivery_bandwidth_bytes_s or config.bandwidth_bytes_s)
            )
        ),
        "configured_delivery_load": config.delivery_load,
        "total_bandwidth_bytes_s": float(capacities.sum())
        if config.bandwidth_mode == "shared"
        else sum(link.bandwidth_bytes_s for link in links),
        "expected_backhaul_load_cold": float(
            effective_rate
            * (sizes * popularity).sum()
            / (
                capacities.sum()
                if config.bandwidth_mode == "shared"
                else config.caches * (config.backhaul_bandwidth_bytes_s or config.bandwidth_bytes_s)
            )
        ),
        "request_count": len(requests),
        "seed": seed,
        "synthetic": True,
    }
    if config.episode_loads:
        metadata["load_phases"] = [
            dict(start_s=start, end_s=end, load=load, arrival_rate=rate)
            for start, end, load, rate in config.load_phases()
        ]
    return run, metadata
