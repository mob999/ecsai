"""Fixed-slot DD adaptation: visibility, timing, mapping and learning closure."""

import json

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.env import OBS, SchedulingEnv
from edge_sim_learning.experiment import evaluate, train
from edge_sim_learning.model import HistoryActor
from edge_sim_learning.scenario import ScenarioConfig, build_run
from edge_sim_learning.torch_env import ContentBatchEnv
from pettingzoo.test import parallel_api_test
from torchrl.envs.utils import check_env_specs


def config():
    return ScenarioConfig.profile("smoke").model_copy(
        update={"scheduler_release": "window", "scheduler_capacity": 8}
    )


def actions(env, forward=False):
    return {
        a: np.array([1 if forward else -1] * env.slots + [0.5], np.float32)
        for a in env.possible_agents
    }


def test_dd_contract_visible_slots_reset_and_drain():
    env = SchedulingEnv(config(), method="DD-adapted")
    try:
        parallel_api_test(env, num_cycles=10)
        initial, _ = env.reset(seed=42)
        assert all(not value.any() for value in initial.values())
        obs, _, _, _, _ = env.step(actions(env))
        assert env.last_metrics["arrived"] > 0
        assert env.last_metrics["cache_hit_rate"] == 0
        assert env.last_view.cache_lookups == 0  # future arrivals cannot be pre-decided
        slots = env._request_slots()
        control = env.control(actions(env, True))
        for a, item in zip(env.possible_agents, control.schedulers, strict=True):
            assert set(item.request_decisions) == {
                r.request_id for r in slots[a] if not r.forwarded
            }
            assert all(item.request_decisions.values())
            features = obs[a][OBS:].reshape(env.slots, 4)
            assert features[:, 3].sum() == len(slots[a])
            for i, request in enumerate(slots[a]):
                assert request.arrival_s <= env.elapsed
                assert features[i, 0] == pytest.approx(request.size_bytes / env.config.max_size)
        while env.agents:
            env.step(actions(env))
        assert env.metrics()["unfinished"] > 0
        drained = env.drain(lambda obs: actions(env))
        assert drained["unfinished"] == 0
        assert (
            drained["arrived"] == drained["completed"] + drained["timed_out"] + drained["rejected"]
        )
        obs, _ = env.reset(seed=42)
        assert all(not value.any() for value in obs.values())
    finally:
        env.close()
    batch = ContentBatchEnv(config(), workers=2, method="DD-adapted")
    try:
        check_env_specs(batch)
        td = batch.rollout(config().cycles, break_when_any_done=False)
        assert td["next", "truncated"][:, -1].all()
        assert not td["next", "terminated"].any()
    finally:
        batch.close()


@pytest.mark.parametrize("forward", [False, True])
def test_dd_matches_threshold_with_same_batch_timing(forward):
    cfg = config()
    dd = SchedulingEnv(cfg, method="DD-adapted")
    threshold = SchedulingEnv(cfg, method="MAPPO-no-context")
    try:
        dd.reset(seed=73)
        threshold.reset(seed=73)
        assert dd.run == threshold.run
        for _ in range(cfg.cycles):
            dd.step(actions(dd, forward))
            threshold.step(
                {
                    a: np.array([0, 0, 0, 5 if forward else -5, 0.5], np.float32)
                    for a in threshold.possible_agents
                }
            )
            assert dd.metrics() == threshold.metrics()
        assert dd.drain(lambda obs: actions(dd, forward)) == threshold.drain()
    finally:
        dd.close()
        threshold.close()
    continuous, _ = build_run(cfg.model_copy(update={"scheduler_release": "continuous"}), 73)
    window, _ = build_run(cfg, 73)
    assert continuous.content.requests == window.content.requests
    assert continuous.scenario == window.scenario


def test_padding_and_forwarded_slots_have_fixed_distribution_and_zero_gradient():
    slots = 3
    actor = HistoryActor(output_dim=2 * (slots + 1), use_context=False, input_dim=OBS + 4 * slots)
    x = torch.zeros(1, OBS + 4 * slots)
    x[0, OBS : OBS + 4] = torch.tensor([0.2, 0.9, 0, 1])
    x[0, OBS + 4 : OBS + 8] = torch.tensor([0.4, 0.5, 1, 1])
    result = actor(x)
    result.retain_grad()
    result[0, [1, 2, 5, 6]].sum().backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in actor.parameters())
    actor.zero_grad()
    actor(x)[0, [0, 3, 4, 7]].sum().backward()
    assert actor.mlp[0].weight.grad.abs().sum() > 0


def test_dd_cpu_update_resume_and_parallel_evaluation(tmp_path):
    cfg = config()
    options = dict(
        scenario=cfg,
        method="DD-adapted",
        workers=1,
        frames_per_batch=8,
        epochs=1,
        minibatch=8,
        eval_interval=8,
        eval_episodes=1,
        hidden_size=32,
    )
    train(output=tmp_path / "first", episodes=1, **options)
    path = tmp_path / "first/last.pt"
    first = torch.load(path, weights_only=False)
    assert first["method"] == "DD-adapted"
    assert first["action_dim"] == cfg.scheduler_capacity + 2
    assert first["scenario"]["scheduler_release"] == "window"
    for mode in ["deterministic", "stochastic"]:
        serial = evaluate(cfg, "DD-adapted", first["policy"], [100, 101], exploration=mode)
        parallel = evaluate(
            cfg, "DD-adapted", first["policy"], [100, 101], exploration=mode, workers=2
        )
        for a, b in zip(serial["episodes"], parallel["episodes"], strict=True):
            assert {k: v for k, v in a.items() if not k.endswith("wall_s")} == {
                k: v for k, v in b.items() if not k.endswith("wall_s")
            }
            assert a["unfinished"] == 0
    train(output=tmp_path / "resumed", episodes=2, resume=path, **options)
    restored = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert restored["experiment"]["state"]["total_frames"] == 16
    assert all(torch.isfinite(p).all() for p in restored["policy"].parameters())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(first["policy"].parameters(), restored["policy"].parameters(), strict=True)
    )
    assert (tmp_path / "first/best.pt").exists()
    assert list((tmp_path / "first/wandb").glob("offline-run-*/*.wandb"))
    evaluation = json.loads((tmp_path / "first/evaluation-8.json").read_text())
    assert evaluation["mean"]["unfinished"] == 0


def test_direct_control_rejects_future_missing_and_wrong_cluster_ids():
    from edge_sim import SDKError
    from edge_sim_models import WindowControl

    env = SchedulingEnv(config(), method="DD-adapted")
    try:
        env.reset(seed=42)
        env.step(actions(env))
        valid = env.control(actions(env))
        index = next(i for i, item in enumerate(valid.schedulers) if item.request_decisions)
        for replacement in ({}, {"future-request": True}):
            items = list(valid.schedulers)
            items[index] = items[index].model_copy(update={"request_decisions": replacement})
            with pytest.raises(SDKError, match="invalid_control"):
                env.session.advance_window(
                    0.2, WindowControl(policy="direct", schedulers=tuple(items))
                )
            assert env.session.inspect().now_s == env.elapsed
        items = list(valid.schedulers)
        items[0], items[1] = (
            items[0].model_copy(
                update={"request_decisions": valid.schedulers[1].request_decisions}
            ),
            items[1].model_copy(
                update={"request_decisions": valid.schedulers[0].request_decisions}
            ),
        )
        with pytest.raises(SDKError, match="invalid_control"):
            env.session.advance_window(0.2, WindowControl(policy="direct", schedulers=tuple(items)))
        env.step(actions(env))  # rejected commands do not corrupt the worker
    finally:
        env.close()


def test_window_threshold_drain_refreshes_policy_and_history():
    env = SchedulingEnv(config(), method="DEPPO-adapted")
    calls = []
    try:
        obs, _ = env.reset(seed=42)
        action = {a: np.array([0, 0, 0, -5, 0.5], np.float32) for a in env.possible_agents}
        while env.agents:
            obs, _, _, _, _ = env.step(action)

        def infer(observations):
            calls.append(observations)
            return action

        assert env.drain(infer)["unfinished"] == 0
        assert calls
        assert env.agents == []
    finally:
        env.close()
