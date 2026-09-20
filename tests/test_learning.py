"""Learning package contracts (optional to the engine-only SDK installation)."""

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.env import FEATURES, OBS, SchedulingEnv
from edge_sim_learning.model import ContextModel, HistoryActor
from edge_sim_learning.scenario import ScenarioConfig, build_run
from edge_sim_learning.torch_env import ContentBatchEnv
from pettingzoo.test import parallel_api_test
from torchrl.envs.utils import check_env_specs


def test_independent_workload_streams_and_profiles():
    cfg = ScenarioConfig.profile("smoke")
    first, meta = build_run(cfg, 42)
    second, _ = build_run(cfg, 42)
    assert first == second
    changed, _ = build_run(cfg.model_copy(update={"caches": 4}), 42)
    assert first.content.requests == changed.content.requests
    assert first.scenario.artifacts == changed.scenario.artifacts
    assert meta["expected_delivery_load"] > 0
    assert [
        (
            ScenarioConfig.profile(k).clusters,
            ScenarioConfig.profile(k).caches,
            ScenarioConfig.profile(k).request_rate,
        )
        for k in ("small", "medium", "large")
    ] == [(3, 10, 300), (5, 20, 1000), (7, 30, 2000)]


def test_parallel_api_history_reward_and_reset():
    cfg = ScenarioConfig.profile("smoke")
    env = SchedulingEnv(cfg)
    try:
        parallel_api_test(env, num_cycles=10)
        obs, _ = env.reset(seed=3)
        actions = {a: np.array([0, 0, 0, -5], np.float32) for a in env.agents}
        assert all(not o.any() for o in obs.values())
        env.begin(actions)
        response = env.response
        n = response.completed + response.timed_out + response.rejected
        expected = 0.5 * (response.completed / n if n else 0) + 0.5 * sum(
            link.bytes_sent for link in response.link_bytes
        ) / sum(link.capacity_byte_seconds for link in response.link_bytes)
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert all(r == pytest.approx(expected) for r in reward.values())
        for a, o in obs.items():
            assert o[-1] == 1
            assert np.array_equal(o[OBS + OBS : OBS + OBS + 4], actions[a])
        for _ in range(cfg.cycles - 1):
            obs, _, terminated, truncated, _ = env.step(actions)
        assert not any(terminated.values()) and all(truncated.values())
        assert env.drain()["unfinished"] == 0
        obs, _ = env.reset(seed=3)
        assert all(not o.any() for o in obs.values())
    finally:
        env.close()


def test_torchrl_contract_and_truncation_bootstrap():
    env = ContentBatchEnv(ScenarioConfig.profile("smoke"), 2)
    try:
        check_env_specs(env)
        td = env.rollout(8, break_when_any_done=False)
        assert td["next", "truncated"][:, -1].all()
        assert not td["next", "terminated"].any()
        assert td["agents", "observation"].shape[-2:] == (2, FEATURES)
    finally:
        env.close()
    from torchrl.objectives.value.functional import generalized_advantage_estimate

    advantage, _ = generalized_advantage_estimate(
        0.99,
        0.95,
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[True]]),
        terminated=torch.tensor([[False]]),
    )
    assert advantage.item() == pytest.approx(1 + 0.99 * 3 - 2)


def test_history_padding_no_context_and_gradients():
    torch.manual_seed(3)
    actor = HistoryActor()
    x = torch.randn(4, FEATURES)
    x[:, -1] = 0
    empty = actor(x)
    x[:, OBS:-1] = 100
    assert torch.equal(empty, actor(x))
    x[:, -1] = 3
    x[:, OBS:-1] = torch.randn_like(x[:, OBS:-1])
    y = actor(x)
    x2 = x.clone()
    x2[:, OBS + 3 * 17 : -1] = 100
    assert torch.equal(y, actor(x2))
    y.sum().backward()
    assert actor.gru.weight_ih_l0.grad.abs().sum() > 0
    baseline = HistoryActor(use_context=False)
    x2[:, OBS:] = 100
    assert torch.equal(baseline(x), baseline(x2))


def test_cpu_update_checkpoint_restore_and_offline_logs(tmp_path):
    from edge_sim_learning.experiment import train

    cfg = ScenarioConfig.profile("smoke")
    common = dict(
        scenario=cfg,
        episodes=2,
        workers=1,
        frames_per_batch=16,
        epochs=1,
        minibatch=8,
        eval_interval=16,
        eval_episodes=1,
    )
    train(output=tmp_path / "first", **common)
    payload = torch.load(tmp_path / "first" / "last.pt", weights_only=False)
    actor = next(m for m in payload["policy"].modules() if isinstance(m, ContextModel))
    assert len(actor.actors) == 2
    from tensordict import TensorDict

    features = torch.randn(2, FEATURES)
    features[:, -1] = 3
    first = (
        actor(TensorDict({actor.in_key: features.clone(), "state": torch.zeros(26)}, []))[
            actor.out_key
        ]
        .detach()
        .clone()
    )
    features[0, :OBS] += 10
    second = actor(TensorDict({actor.in_key: features, "state": torch.ones(26)}, []))[
        actor.out_key
    ].detach()
    assert torch.equal(first[1], second[1])  # neither another agent nor global state leaks
    assert not torch.equal(first[0], second[0])
    assert not (
        {p.data_ptr() for p in actor.actors[0].parameters()}
        & {p.data_ptr() for p in actor.actors[1].parameters()}
    )
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    loss = payload["experiment"]["loss_agents"]
    critic_keys = [k for k in loss if "critic_network_params" in k and k.endswith("weight")]
    assert critic_keys
    # A single 26 -> 128 centralized critic, without an agent parameter axis.
    assert any(tuple(loss[k].shape) == (128, 26) for k in critic_keys)
    assert payload["optimizers"]["agents"]
    moments = [
        state["exp_avg"]
        for optimizer in payload["optimizers"]["agents"].values()
        for state in optimizer["state"].values()
        if "exp_avg" in state
    ]
    assert any(tuple(moment.shape) == (192, 17) and moment.abs().sum() > 0 for moment in moments)
    common["episodes"] = 4
    train(output=tmp_path / "restored", resume=tmp_path / "first" / "last.pt", **common)
    restored = torch.load(tmp_path / "restored" / "last.pt", weights_only=False)
    assert restored["experiment"]["state"]["total_frames"] == 32
    assert (tmp_path / "restored" / "best.pt").exists()
    assert any(
        not torch.equal(v, restored["experiment"]["loss_agents"][k])
        for k, v in loss.items()
        if torch.is_tensor(v)
    )
    csv = (tmp_path / "first" / "metrics.csv").read_text()
    for key in (
        "train/reward",
        "train/episode_return",
        "train/loss_objective",
        "train/loss_critic",
        "train/entropy",
        "train/inference_wall_s",
        "eval/success_rate",
        "eval/latency_p99_s",
    ):
        assert key in csv
    assert list((tmp_path / "first" / "wandb").glob("offline-run-*/*.wandb"))
    assert (tmp_path / "first" / "best.pt").exists()


def test_backhaul_and_delivery_configured_independently():
    config = ScenarioConfig.profile("smoke").model_copy(
        update={
            "backhaul_bandwidth_bytes_s": 1000,
            "delivery_bandwidth_bytes_s": 2000,
            "backhaul_concurrency": 2,
            "delivery_concurrency": 3,
            "backhaul_waiting": 0,
            "delivery_waiting": 7,
        }
    )
    run, _ = build_run(config, 1)
    links = {link.id: link for link in run.scenario.links}
    for cache in run.content.caches:
        assert links[cache.backhaul_link].bandwidth_bytes_s == 1000
        assert links[cache.delivery_link].bandwidth_bytes_s == 2000
        assert cache.backhaul.max_active == 2 and cache.backhaul.max_waiting == 0
        assert cache.delivery.max_active == 3 and cache.delivery.max_waiting == 7
