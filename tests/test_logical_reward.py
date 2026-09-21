"""Logical rewards reconcile with the retry ledger, without changing the simulation."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
from edge_sim_learning.env import SchedulingEnv
from edge_sim_learning.scenario import ScenarioConfig


def test_logical_reward_settles_once_and_counts_full_elapsed():
    cfg = ScenarioConfig.profile("smoke").model_copy(
        update={"reward_mode": "logical", "max_retries": 2}
    )
    env = SchedulingEnv(cfg)
    try:
        env.reset(seed=0)

        def outcome(original, index, status, end):
            return SimpleNamespace(
                original_request_id=original, attempt_index=index, status=status,
                first_arrival_s=0.0, completed_s=end,
            )

        assert env._settle_logical_reward([outcome("a", 0, "TIMED_OUT", 1)]) == 0
        assert env._settle_logical_reward([outcome("a", 1, "TIMED_OUT", 2.1)]) == 0
        final = outcome("a", 2, "SUCCEEDED", 2.7)
        assert env._settle_logical_reward([final]) == pytest.approx((1 - 0.27) / 3)
        assert env._settle_logical_reward([final]) == 0
        assert env._settle_logical_reward([
            outcome("b", 2, "REJECTED", 0.2), outcome("c", 2, "TIMED_OUT", 3.2)
        ]) == pytest.approx((-2 - 0.34) / 3)
        assert env.logical_completed == 1 and env.logical_failed == 2
        assert env.logical_elapsed_s == pytest.approx(6.1)
        env.reset(seed=0)
        assert not env.logical_settled and env.logical_return == 0
    finally:
        env.close()


def test_logical_reward_reconciles_after_drain_and_preserves_workload():
    cfg = ScenarioConfig.profile("smoke").model_copy(update={
        "max_retries": 2, "deadline_s": 0.1, "cycles": 8,
        "episode_loads": (0.75,), "delivery_load": 0.75,
    })
    results = []
    for mode in ["business", "logical"]:
        env = SchedulingEnv(cfg.model_copy(update={"reward_mode": mode}))
        try:
            env.reset(seed=23)
            actions = {a: np.array([0, 0, 0, -2, 0.5], np.float32) for a in env.agents}
            rewards = []
            while env.agents:
                _, reward, terminated, truncated, _ = env.step(actions)
                rewards.append(next(iter(reward.values())))
            assert all(truncated.values()) and not any(terminated.values())
            train_return = env.episode_return
            metrics = env.drain()
            assert env.episode_return == train_return
            rows = env.retry_records()
            results.append((env.run.content.requests, rows))
            if mode == "logical":
                assert env.logical_completed == metrics["logical_completed"]
                assert env.logical_failed == metrics["logical_failed"]
                assert len(env.logical_settled) == metrics["logical_requests"]
                assert env.logical_elapsed_s == pytest.approx(metrics["total_elapsed_s"])
                expected = (
                    metrics["logical_completed"] - metrics["logical_failed"]
                    - 0.1 * metrics["total_elapsed_s"] / cfg.deadline_s
                ) / env.logical_normalizer
                assert metrics["logical_settled_return"] == pytest.approx(expected)
                assert sum(rewards) == pytest.approx(train_return)
                assert metrics["retry_attempts"] > 0
        finally:
            env.close()
    assert results[0] == results[1]
