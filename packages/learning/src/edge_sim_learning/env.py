"""PettingZoo boundary: observations/rewards live here, physics stays in the SDK."""

from collections import deque
from time import perf_counter

import numpy as np
from edge_sim import start
from edge_sim_models import SchedulerControl, WindowControl
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from .scenario import ScenarioConfig, build_run

HISTORY = 8
OBS = 13
ACTION = 5
FEATURES = OBS + HISTORY * (OBS + ACTION) + 1


class SchedulingEnv(ParallelEnv):
    metadata = {"name": "edge_content_v1", "render_modes": [], "is_parallelizable": True}

    def __init__(self, config=None, seed=0, runner=None, slot=0, evaluation=False):
        self.config = config or ScenarioConfig()
        self.possible_agents = [f"cluster-{i}" for i in range(self.config.clusters)]
        self.agents = []
        self.runner, self.slot, self.seed_value = runner, slot, seed
        self.evaluation = evaluation
        self.generation = 0
        self.session = None
        self.run_id = None
        self.response = None
        self.state_space = Box(-np.inf, np.inf, (OBS * self.config.clusters,), np.float32)
        self.last_metrics = {}
        self._observation_space = Box(-np.inf, np.inf, (FEATURES,), np.float32)
        self._action_space = Box(
            np.array([-5] * 4 + [0.05], np.float32),
            np.array([5] * 4 + [0.95], np.float32),
            dtype=np.float32,
        )

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed
            self.generation = 0
        episode_key = (self.seed_value, self.slot, self.generation)
        # TorchRL probes the environment before the collector's initial reset.
        # An untouched, identical cold episode can serve both; never reuse an
        # advanced episode or a different seed/generation.
        pristine = (
            self.session is not None
            and not self.session.closed
            and self._pristine
            and self.response is None
            and self._episode_key == episode_key
        )
        if not pristine:
            self.close()
        episode_seed = int(
            np.random.SeedSequence([self.seed_value, self.slot, self.generation]).generate_state(1)[
                0
            ]
        )
        self.run_id = f"slot-{self.slot}-episode-{self.generation}"
        self.generation += 1
        if not pristine:
            self.run, self.workload = build_run(self.config, episode_seed, self.run_id)
            if self.runner is None:
                self.session = start(self.run)
            else:
                self.runner.submit(self.run)
                self.session = self.runner.session(self.run_id)
        self._episode_key = episode_key
        self._pristine = True
        self.cache_clusters = {c.node_id: c.cluster_id for c in self.run.content.caches}
        self.link_ids = {
            kind: {getattr(c, kind + "_link") for c in self.run.content.caches}
            for kind in ("backhaul", "delivery")
        }
        self.agents = self.possible_agents.copy()
        self.cycle = 0
        self.elapsed = 0
        self.history = {a: deque(maxlen=HISTORY) for a in self.agents}
        self.current = {a: np.zeros(OBS, np.float32) for a in self.agents}
        self.last_view = self.session.inspect()
        self.latencies = []
        self.episode_return = 0
        self.response = None
        self.response_received = None
        self.last_metrics = self.metrics()
        return self._observations(), {a: {} for a in self.agents}

    def state(self):
        return np.concatenate([self.current[a] for a in self.possible_agents]).astype(np.float32)

    def _observations(self):
        result = {}
        for a in self.possible_agents:
            value = np.zeros(FEATURES, np.float32)
            value[:OBS] = self.current[a]
            history = value[OBS:-1].reshape(HISTORY, OBS + ACTION)
            for i, pair in enumerate(self.history[a]):
                history[i] = pair
            value[-1] = len(self.history[a])
            result[a] = value
        return result

    def _encode(self, view):
        requests = {r.request_id: r for r in view.requests}
        clustered_pools = {a: [] for a in self.possible_agents}
        for pool in view.pools:
            clustered_pools[self.cache_clusters[pool.node_id]].append(pool)
        for scheduler in view.schedulers:
            pools = clustered_pools[scheduler.cluster_id]
            loads = {}
            for kind in ("backhaul", "delivery"):
                values = [
                    (p.active + p.waiting) / max(1, (p.max_active or 1) + (p.max_waiting or 0))
                    for p in pools
                    if p.kind == kind
                ]
                loads[kind] = sum(values) / len(values)
            waiting = [requests[r] for r in scheduler.waiting]
            if not waiting:
                value = np.zeros(OBS, np.float32)
                value[1:3] = loads["backhaul"], loads["delivery"]
                self.current[scheduler.cluster_id] = value
                continue
            size_bins = np.histogram(
                [min(1, r.size_bytes / self.config.max_size) for r in waiting], bins=5, range=(0, 1)
            )[0]
            time_bins = np.histogram(
                [
                    min(1, max(0, r.deadline_s - view.now_s) / self.config.deadline_s)
                    for r in waiting
                ],
                bins=5,
                range=(0, 1),
            )[0]
            self.current[scheduler.cluster_id] = np.concatenate(
                (
                    [
                        len(waiting) / max(1, scheduler.capacity),
                        loads["backhaul"],
                        loads["delivery"],
                    ],
                    size_bins / max(1, len(waiting)),
                    time_bins / max(1, len(waiting)),
                )
            ).astype(np.float32)

    def control(self, actions, policy="threshold"):
        if set(actions) != set(self.agents):
            raise ValueError("one action per live agent is required")
        for a, action in actions.items():
            if not self.action_space(a).contains(np.asarray(action, dtype=np.float32)):
                raise ValueError(f"invalid action for {a}")
        return WindowControl(
            policy=policy,
            schedulers=tuple(
                SchedulerControl(
                    cluster_id=a,
                    weights=tuple(float(v) for v in actions[a][:4]),
                    backhaul_ratio=float(np.clip(actions[a][4], 0.05, 0.95)),
                )
                for a in self.possible_agents
            ),
        )

    def begin(self, actions, policy="threshold"):
        self.pending_control = self.control(actions, policy)
        self._pristine = False
        self.pending_actions = {
            a: np.asarray(v, dtype=np.float32).copy() for a, v in actions.items()
        }
        self.step_started = perf_counter()
        self.response_received = None
        boundary = (self.cycle + 1) * self.config.period_s
        if self.runner is not None:
            self.runner.submit_window(
                self.run_id, boundary, self.pending_control, scope="scheduling"
            )
        else:
            self.response = self.session.advance_window(
                boundary, self.pending_control, scope="scheduling"
            )
            self.response_received = perf_counter()

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if self.response is None:
            self.begin(actions)
        if self.runner is not None and self.response is None:
            raise RuntimeError("batched environment must collect submitted responses")
        response, self.response = self.response, None
        encoding_started = perf_counter()
        wall = (self.response_received or encoding_started) - self.step_started
        for a in self.agents:
            self.history[a].append(np.concatenate((self.current[a], self.pending_actions[a])))
        self.cycle += 1
        self.last_view = response.view
        self._encode(response.view)
        self.latencies.extend(response.view.latencies_s)
        self.elapsed = response.view.now_s
        resolved = response.completed + response.timed_out + response.rejected
        success = response.completed / resolved if resolved else 0
        capacity = sum(link.capacity_byte_seconds for link in response.link_bytes)
        utilization = (
            sum(link.bytes_sent for link in response.link_bytes) / capacity if capacity else 0
        )
        paper_reward = 0.5 * success + 0.5 * utilization
        business_reward = (
            response.completed
            - response.timed_out
            - response.rejected
            - 0.1 * sum(response.view.latencies_s) / self.config.deadline_s
        ) / max(1, self.config.request_rate * self.config.period_s)
        reward = paper_reward if self.config.reward_mode == "paper" else business_reward
        self.episode_return += reward
        done = self.cycle >= self.config.cycles
        self.last_metrics = self.metrics() | {
            "reward": reward,
            "paper_reward": paper_reward,
            "business_reward": business_reward,
            "episode_return": self.episode_return,
            "simulation_wall_s": response.simulation_wall_s,
            "ipc_wall_s": max(0, wall - response.simulation_wall_s),
            "rpc_overhead_wall_s": max(0, wall - response.simulation_wall_s),
            "window_wall_s": wall,
            "window_completed": response.completed,
            "window_resolved": resolved,
        }
        observations = self._observations()
        self.last_metrics["encoding_wall_s"] = perf_counter() - encoding_started
        rewards = dict.fromkeys(self.agents, reward)
        terminated = dict.fromkeys(self.agents, False)
        truncated = dict.fromkeys(self.agents, done)
        infos = {a: self.last_metrics.copy() for a in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    def drain(self):
        """Evaluation only: no new arrivals, fixed last control, no extra learner steps."""
        limit = self.config.horizon_s + self.config.deadline_s
        while self.elapsed < limit:
            response = self.session.advance_window(
                min(limit, self.elapsed + self.config.period_s),
                self.pending_control,
                scope="scheduling",
            )
            self.last_view = response.view
            self.elapsed = response.view.now_s
            self.latencies.extend(response.view.latencies_s)
            if response.kind == "finished":
                break
        return self.metrics()

    def metrics(self):
        v = self.last_view
        resolved = v.completed + v.timed_out + v.rejected
        metrics = {
            "success_rate": v.completed / resolved if resolved else 0,
            "timeout_rate": v.timed_out / resolved if resolved else 0,
            "rejection_rate": v.rejected / resolved if resolved else 0,
            "overflow_rate": v.overflows / resolved if resolved else 0,
            "unfinished": v.arrived - resolved,
            "arrived": v.arrived,
            "completed": v.completed,
            "timed_out": v.timed_out,
            "rejected": v.rejected,
            "cancelled_transfers": v.cancelled_transfers,
            "cache_hit_rate": v.cache_hits / v.cache_lookups if v.cache_lookups else 0,
            "forward_ratio": v.forwarded / v.arrived if v.arrived else 0,
            "mean_latency_s": float(np.mean(self.latencies)) if self.latencies else 0,
            "simulated_time_s": v.now_s,
        }
        percentiles = np.percentile(self.latencies, [50, 95, 99]) if self.latencies else (0, 0, 0)
        for quantile, value in zip((50, 95, 99), percentiles, strict=True):
            metrics[f"latency_p{quantile}_s"] = float(value)
        for kind in ("backhaul", "delivery"):
            ids = self.link_ids[kind]
            counters = [link for link in v.links if link.link_id in ids]
            total = sum(link.bytes_sent for link in counters)
            capacity = sum(link.capacity_byte_seconds for link in counters)
            metrics[kind + "_bytes"] = total
            metrics[kind + "_utilization"] = total / capacity if capacity else 0
            pools = [p for p in v.pools if p.kind == kind]
            metrics[kind + "_waiting"] = sum(p.waiting for p in pools)
            metrics[kind + "_active"] = sum(p.active for p in pools)
        metrics["scheduler_waiting"] = sum(len(s.waiting) for s in v.schedulers)
        return metrics

    def close(self):
        if self.session is not None:
            if self.runner is None:
                self.session.close()
            else:
                self.runner.discard(self.run_id)
            self.session = None
