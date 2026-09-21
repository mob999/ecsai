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

    def __init__(
        self, config=None, seed=0, runner=None, slot=0, evaluation=False, method="DEPPO-adapted"
    ):
        self.config = config or ScenarioConfig()
        if self.config.reward_mode == "logical" and not self.config.max_retries:
            raise ValueError("logical reward requires retry outcome telemetry (max_retries > 0)")
        self.method = method
        self.direct = method == "DD-adapted"
        if self.direct and self.config.scheduler_release != "window":
            raise ValueError("DD-adapted requires scheduler_release=window")
        self.slots = self.config.scheduler_capacity + 1
        self.action_dim = self.slots + 1 if self.direct else ACTION
        self.possible_agents = [f"cluster-{i}" for i in range(self.config.clusters)]
        self.agents = []
        self.runner, self.slot, self.seed_value = runner, slot, seed
        self.evaluation = evaluation
        self.generation = 0
        self.session = None
        self.run_id = None
        self.response = None
        self._prepared_key = None
        self.state_space = Box(-np.inf, np.inf, (OBS * self.config.clusters,), np.float32)
        self.last_metrics = {}
        self._observation_space = Box(-np.inf, np.inf, (FEATURES,), np.float32)
        self._action_space = Box(
            np.array([-5] * 4 + [0.05], np.float32),
            np.array([5] * 4 + [0.95], np.float32),
            dtype=np.float32,
        )

        if self.direct:
            self._observation_space = Box(-np.inf, np.inf, (OBS + self.slots * 4,), np.float32)
            self._action_space = Box(
                np.array([-1] * self.slots + [0.05], np.float32),
                np.array([1] * self.slots + [0.95], np.float32),
            )

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def prepare_reset(self, seed=None):
        """Prepare a workload without booting; batch callers can start all together."""
        if seed is not None:
            self.seed_value, self.generation = seed, 0
        episode_key = (self.seed_value, self.slot, self.generation)
        if self._prepared_key == episode_key:
            return
        pristine = (
            self.session is not None
            and not self.session.closed
            and self._pristine
            and self.response is None
            and self._episode_key == episode_key
        )
        if not pristine:
            self.close()
            episode_seed = int(np.random.SeedSequence(episode_key).generate_state(1)[0])
            self.run_id = f"slot-{self.slot}-episode-{self.generation}"
            self.run, self.workload = build_run(self.config, episode_seed, self.run_id)
        self._prepared_key = episode_key

    def reset(self, seed=None, options=None):
        self.prepare_reset(seed)
        if self.session is None:
            if self.runner is None:
                self.session = start(self.run)
            else:
                self.runner.submit(self.run)
                self.session = self.runner.session(self.run_id)
        self._episode_key = self._prepared_key
        self._prepared_key = None
        self.generation += 1
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
        self.attempt_outcomes = {}
        self.logical_settled = set()
        self.logical_completed = self.logical_failed = 0
        self.logical_elapsed_s = 0.0
        self.logical_return = 0.0
        self.logical_normalizer = max(
            1, self.config.expected_arrivals(0, self.config.horizon_s) / self.config.cycles
        )
        self.episode_return = 0
        self.response = None
        self.response_received = None
        self.last_metrics = self.metrics()
        return self._observations(), {a: {} for a in self.agents}

    def state(self):
        return np.concatenate([self.current[a] for a in self.possible_agents]).astype(np.float32)

    def _request_slots(self):
        requests = {r.request_id: r for r in self.last_view.requests}
        slots = {}
        for scheduler in self.last_view.schedulers:
            ids = ([scheduler.active_request] if scheduler.active_request else []) + list(
                scheduler.waiting
            )
            slots[scheduler.cluster_id] = [requests[rid] for rid in ids]
        return slots

    def _observations(self):
        if self.direct:
            result = {}
            for agent, requests in self._request_slots().items():
                features = np.zeros((self.slots, 4), np.float32)
                for i, r in enumerate(requests):
                    features[i] = (
                        min(1, r.size_bytes / self.config.max_size),
                        np.clip(
                            (r.deadline_s - self.last_view.now_s) / self.config.deadline_s, 0, 1
                        ),
                        float(r.forwarded),
                        1,
                    )
                result[agent] = np.concatenate((self.current[agent], features.ravel()))
            return result
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
        if self.direct:
            if policy != "threshold":
                raise ValueError("DD-adapted uses explicit per-request decisions")
            slots = self._request_slots()
            return WindowControl(
                policy="direct",
                schedulers=tuple(
                    SchedulerControl(
                        cluster_id=a,
                        backhaul_ratio=float(actions[a][-1]),
                        request_decisions={
                            r.request_id: bool(actions[a][i] >= 0)
                            for i, r in enumerate(slots[a])
                            if not r.forwarded
                        },
                    )
                    for a in self.possible_agents
                ),
            )
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
        for a in [] if self.direct else self.agents:
            self.history[a].append(np.concatenate((self.current[a], self.pending_actions[a])))
        self.cycle += 1
        self.last_view = response.view
        self.attempt_outcomes.update({r.request_id: r for r in response.view.attempt_outcomes})
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
        ) / max(
            1,
            self.config.expected_arrivals(self.elapsed - self.config.period_s, self.elapsed)
            if self.config.episode_loads
            else self.config.request_rate * self.config.period_s,
        )
        logical_reward = (
            self._settle_logical_reward(response.view.attempt_outcomes)
            if self.config.reward_mode == "logical"
            else 0.0
        )
        reward = {"paper": paper_reward, "business": business_reward, "logical": logical_reward}[
            self.config.reward_mode
        ]
        reward *= self.config.reward_scale
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
        if self.config.reward_mode == "logical":
            self.last_metrics["logical_reward"] = logical_reward
        observations = self._observations()
        self.last_metrics["encoding_wall_s"] = perf_counter() - encoding_started
        rewards = dict.fromkeys(self.agents, reward)
        terminated = dict.fromkeys(self.agents, False)
        truncated = dict.fromkeys(self.agents, done)
        infos = {a: self.last_metrics.copy() for a in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    def _settle_logical_reward(self, outcomes):
        """Settle each original once, including all failed attempts and retry waits."""
        numerator = 0.0
        for outcome in outcomes:
            succeeded = outcome.status == "SUCCEEDED"
            exhausted = (
                outcome.status in {"TIMED_OUT", "REJECTED"}
                and outcome.attempt_index == self.config.max_retries
            )
            if not (succeeded or exhausted):
                continue
            original = outcome.original_request_id
            if original in self.logical_settled:
                continue
            elapsed = outcome.completed_s - outcome.first_arrival_s
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid logical request elapsed time")
            self.logical_settled.add(original)
            self.logical_completed += int(succeeded)
            self.logical_failed += int(not succeeded)
            self.logical_elapsed_s += elapsed
            numerator += (1 if succeeded else -1) - 0.1 * elapsed / self.config.deadline_s
        reward = numerator / self.logical_normalizer
        self.logical_return += reward
        return reward

    def drain(self, action_fn=None, policy="threshold"):
        """Drain without learner steps; batch policies decide newly visible requests."""
        limit = (
            self.config.horizon_s
            + (self.config.max_retries + 1) * self.config.deadline_s
            + self.config.max_retries * self.config.retry_delay_s
        )
        while self.elapsed < limit:
            if self.direct and action_fn is None:
                raise ValueError("DD drain requires policy inference for newly visible requests")
            if self.config.scheduler_release == "window" and action_fn is not None:
                # No training transition or reward is added during evaluation drain.
                live_agents, self.agents = self.agents, self.possible_agents.copy()
                try:
                    actions = action_fn(self._observations())
                    self.pending_control = self.control(actions, policy=policy)
                    if not self.direct:
                        for a in self.possible_agents:
                            self.history[a].append(np.concatenate((self.current[a], actions[a])))
                finally:
                    self.agents = live_agents
            response = self.session.advance_window(
                min(limit, self.elapsed + self.config.period_s),
                self.pending_control,
                scope="scheduling",
            )
            self.last_view = response.view
            self.attempt_outcomes.update({r.request_id: r for r in response.view.attempt_outcomes})
            if self.config.reward_mode == "logical":
                # Report the full ledger separately; drain adds no PPO transitions.
                self._settle_logical_reward(response.view.attempt_outcomes)
            self._encode(response.view)
            self.elapsed = response.view.now_s
            self.latencies.extend(response.view.latencies_s)
            if response.kind == "finished":
                break
        metrics = self.metrics()
        if self.config.max_retries:
            metrics.update(self.retry_metrics())
        return metrics

    def retry_records(self):
        groups = {}
        for attempt in self.attempt_outcomes.values():
            groups.setdefault(attempt.original_request_id, []).append(attempt)
        records = []
        for original in self.run.content.requests:
            attempts = sorted(groups.get(original.id, []), key=lambda r: r.attempt_index)
            if not attempts:
                raise ValueError("missing request outcome after retry drain")
            last = attempts[-1]
            if last.status != "SUCCEEDED" and last.attempt_index != self.config.max_retries:
                raise ValueError("retry drain ended before all attempts resolved")
            if [a.attempt_index for a in attempts] != list(range(len(attempts))):
                raise ValueError("missing attempt in retry trajectory")
            total = last.completed_s - original.arrival_s
            service = sum(a.completed_s - a.arrival_s for a in attempts)
            wait = self.config.retry_delay_s * (len(attempts) - 1)
            if not np.isclose(total, service + wait, atol=1e-7, rtol=1e-7):
                raise ValueError("retry end-to-end duration does not reconcile")
            records.append(
                {
                    "request_id": original.id,
                    "status": last.status,
                    "attempts": len(attempts),
                    "total_elapsed_s": total,
                    "attempt_time_s": service,
                    "retry_wait_s": wait,
                    "attempt_outcomes": [a.model_dump() for a in attempts],
                }
            )
        return records

    def retry_metrics(self):
        records = self.retry_records()
        succeeded = [r for r in records if r["status"] == "SUCCEEDED"]
        failed = [r for r in records if r["status"] != "SUCCEEDED"]

        def average(rows):
            return float(np.mean([r["total_elapsed_s"] for r in rows])) if rows else 0.0

        return {
            "logical_requests": len(records),
            "logical_completed": len(succeeded),
            "logical_failed": len(failed),
            "logical_unfinished": 0,
            "logical_success_rate": len(succeeded) / max(1, len(records)),
            # Failure is terminal observation, not successful delivery.
            "mean_resolution_time_s": average(records),
            "mean_success_e2e_s": average(succeeded),
            "mean_failed_elapsed_s": average(failed),
            "retry_attempts": sum(r["attempts"] - 1 for r in records),
            "mean_attempts": sum(r["attempts"] for r in records) / max(1, len(records)),
            "total_elapsed_s": sum(r["total_elapsed_s"] for r in records),
            "success_e2e_p95_s": float(np.percentile([r["total_elapsed_s"] for r in succeeded], 95))
            if succeeded
            else 0.0,
        }

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
        if self.config.reward_mode == "logical":
            metrics.update(
                logical_settled_completed=self.logical_completed,
                logical_settled_failed=self.logical_failed,
                logical_settled_elapsed_s=self.logical_elapsed_s,
                logical_settled_return=self.logical_return,
            )
        return metrics

    def close(self):
        if self.session is not None:
            if self.runner is None:
                self.session.close()
            else:
                self.runner.discard(self.run_id)
            self.session = None
