"""Versioned local observations; no future workload or remote dynamic state."""

from collections import deque

import numpy as np
from gymnasium.spaces import Box

from .env import OBS, SchedulingEnv

PROFILE = "local-context-v2"
CONTEXT_DIM = OBS + 8
CONTEXT_NAMES = (
    "log_clusters",
    "log_local_caches",
    "log_local_capacity_MBps",
    "log_local_cache_MB",
    "arrival_bytes_per_capacity_second",
    "backhaul_backlog_capacity_seconds",
    "delivery_backlog_capacity_seconds",
    "arrival_window_fill",
)


class LocalContextEnv(SchedulingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.direct:
            raise ValueError("local context is not a DD request-slot observation")
        self._observation_space = Box(-np.inf, np.inf, (CONTEXT_DIM,), np.float32)
        self._reset_context()

    def _reset_context(self):
        self.arrival_history = {a: deque() for a in self.possible_agents}
        self.context_clock = -1.0
        self.resources = None

    def reset(self, seed=None, options=None):
        self._reset_context()
        return super().reset(seed, options)

    def prepare_reset(self, seed=None):
        super().prepare_reset(seed)
        if not self.run.content.report_arrivals:
            self.run = self.run.model_copy(
                update={"content": self.run.content.model_copy(update={"report_arrivals": True})}
            )

    def _encode(self, view):
        super()._encode(view)
        if view.now_s != self.context_clock:
            for arrival in view.window_arrivals:
                if arrival.arrival_s > view.now_s:
                    raise ValueError("future arrival in local telemetry")
                self.arrival_history[arrival.cluster_id].append(
                    (arrival.arrival_s, arrival.size_bytes)
                )
            for history in self.arrival_history.values():
                while history and history[0][0] < view.now_s - 1.0 - 1e-9:
                    history.popleft()
            self.context_clock = view.now_s

    def _observations(self):
        if self.resources is None:
            nodes = {n.id: n for n in self.run.scenario.nodes}
            links = {link.id: link for link in self.run.scenario.links}
            self.resources = {}
            for agent in self.possible_agents:
                caches = [c for c in self.run.content.caches if c.cluster_id == agent]
                capacity = sum(
                    c.total_bandwidth_bytes_s
                    or (
                        links[c.backhaul_link].bandwidth_bytes_s
                        + links[c.delivery_link].bandwidth_bytes_s
                    )
                    for c in caches
                )
                static = np.log1p(
                    [
                        self.config.clusters,
                        len(caches),
                        capacity / 1e6,
                        sum(nodes[c.node_id].storage_bytes for c in caches) / 1e6,
                    ]
                )
                self.resources[agent] = capacity, static
        result = {}
        window = min(1.0, self.last_view.now_s)
        for agent in self.possible_agents:
            capacity, static = self.resources[agent]
            back = delivery = 0.0
            for pool in self.last_view.pools:
                if self.cache_clusters[pool.node_id] == agent:
                    if pool.kind == "backhaul":
                        back += pool.remaining_bytes
                    else:
                        delivery += pool.remaining_bytes
            rate = sum(b for _, b in self.arrival_history[agent]) / window if window else 0.0
            result[agent] = np.concatenate(
                (
                    self.current[agent],
                    static,
                    [
                        rate / capacity,
                        back / capacity,
                        delivery / capacity,
                        window,
                    ],
                )
            ).astype(np.float32)
        return result
