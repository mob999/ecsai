"""Learning package contracts (optional to the engine-only SDK installation)."""

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.env import ACTION, FEATURES, OBS, SchedulingEnv
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
        actions = {a: np.array([0, 0, 0, -5, 0.5], np.float32) for a in env.agents}
        assert all(not o.any() for o in obs.values())
        env.begin(actions)
        response = env.response
        expected = (
            response.completed
            - response.timed_out
            - response.rejected
            - 0.1 * sum(response.view.latencies_s) / cfg.deadline_s
        ) / max(1, cfg.request_rate * cfg.period_s)
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert all(r == pytest.approx(expected) for r in reward.values())
        for a, o in obs.items():
            assert o[-1] == 1
            assert np.array_equal(o[OBS + OBS : OBS + OBS + ACTION], actions[a])
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
    x2[:, OBS + 3 * (OBS + ACTION) : -1] = 100
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
    assert any(
        tuple(moment.shape) == (192, OBS + ACTION) and moment.abs().sum() > 0 for moment in moments
    )
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
            "bandwidth_mode": "independent",
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


@pytest.mark.parametrize("episode_start", [0, 3])
def test_spec_probe_reuses_only_identical_pristine_episode(episode_start):
    env = ContentBatchEnv(ScenarioConfig.profile("smoke"), 1, seed=42, episode_start=episode_start)
    try:
        probe = env.envs[0]
        pid = probe.session.pid
        workload = probe.run
        td = env.reset()
        assert probe.session.pid == pid
        assert probe.run == workload
        assert probe.generation == episode_start + 1
        env.step(env.rand_action(td))
        env.set_seed(42)
        env.reset()
        assert probe.session.pid != pid  # used cache/history cannot be reused
        if episode_start == 0:
            assert probe.run == workload
        else:
            assert probe.run != workload
        assert probe.generation == 1
    finally:
        env.close()


def test_ready_batch_slots_match_independent_episodes():
    config = ScenarioConfig.profile("smoke")
    batch = ContentBatchEnv(config, 2, seed=13)
    singles = [SchedulingEnv(config, seed=13, slot=i) for i in range(2)]
    try:
        td = batch.reset()
        for env in singles:
            env.reset()
        for _ in range(config.cycles):
            actions = torch.tensor(
                [[[0, 0, 0, -5, 0.5]] * 2, [[0, 0, 0, 5, 0.5]] * 2], dtype=torch.float32
            )
            td["agents", "action"] = actions
            following = batch.step(td)["next"]
            for i, env in enumerate(singles):
                obs, rewards, _, truncated, _ = env.step(
                    dict(zip(env.possible_agents, actions[i].numpy(), strict=True))
                )
                np.testing.assert_array_equal(
                    following["agents", "observation"][i].numpy(), np.stack(list(obs.values()))
                )
                np.testing.assert_allclose(
                    following["agents", "reward"][i].numpy().ravel(), list(rewards.values())
                )
                assert following["truncated"][i].item() == all(truncated.values())
                assert batch.envs[i].last_metrics["arrived"] == env.last_metrics["arrived"]
            td = following.exclude("reward", ("agents", "reward"))
    finally:
        batch.close()
        for env in singles:
            env.close()


def test_restore_cuda_rng_normalizes_checkpoint_states_to_cpu(monkeypatch):
    import random
    from types import SimpleNamespace

    from edge_sim_learning.experiment import restore_rng_states

    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    received = []
    payload = {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [SimpleNamespace(cpu=lambda: state)],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    expected = torch.rand(3), np.random.random(3), random.random()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", received.extend)
    restore_rng_states(payload)
    assert len(received) == 1 and received[0] is state
    assert torch.equal(torch.rand(3), expected[0])
    np.testing.assert_array_equal(np.random.random(3), expected[1])
    assert random.random() == expected[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA GPU")
def test_cuda_checkpoint_resume(tmp_path):
    from edge_sim_learning.experiment import train

    config = ScenarioConfig.profile("smoke")
    options = dict(
        scenario=config,
        workers=1,
        frames_per_batch=8,
        epochs=1,
        minibatch=8,
        eval_interval=8,
        eval_episodes=1,
        device="cuda",
    )
    train(output=tmp_path / "first", episodes=1, **options)
    train(output=tmp_path / "resumed", episodes=2, resume=tmp_path / "first/last.pt", **options)
    payload = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=False)
    assert payload["experiment"]["state"]["total_frames"] == 16
    assert payload["cuda_rng"]
    assert all(torch.isfinite(p).all() for p in payload["policy"].parameters())


def test_capacity_profiles_and_fixed_topology():
    cfg = ScenarioConfig.profile("small")
    first, meta = build_run(cfg, 0)
    second, _ = build_run(cfg, 1)
    assert first.content.caches == second.content.caches
    assert first.scenario.links == second.scenario.links
    assert sum(c.total_bandwidth_bytes_s for c in first.content.caches) == pytest.approx(
        cfg.request_rate * (cfg.min_size + cfg.max_size) / 2 / cfg.delivery_load
    )
    assert all(c.backhaul.max_active == 1 for c in first.content.caches)
    assert not first.content.coalesce_backhaul
    audit, _ = build_run(cfg.model_copy(update={"capacity_profile": "paper-audit"}), 0)
    assert all(12e6 / 8 <= c.total_bandwidth_bytes_s <= 36e6 / 8 for c in audit.content.caches)
    assert meta["expected_backhaul_load_cold"] == meta["expected_delivery_load"]


def test_bounded_distribution_extremes_and_probability_recomputation():
    from tensordict.nn import NormalParamExtractor
    from torchrl.modules import TanhNormal

    model = HistoryActor()
    x = torch.full((32, FEATURES), 1e6)
    x[:, -1] = 8
    loc, scale = NormalParamExtractor(scale_mapping="biased_softplus_1.0")(model(x))
    assert loc.abs().max() <= 3
    assert scale.min() > 0 and scale.max() < 2
    low, high = torch.tensor([-5.0] * 4 + [0.05]), torch.tensor([5.0] * 4 + [0.95])
    dist = TanhNormal(loc, scale, low=low, high=high, safe_tanh=True)
    action = dist.sample()
    lp = dist.log_prob(action)
    assert torch.isfinite(lp).all()
    assert torch.equal(lp, TanhNormal(loc, scale, low=low, high=high).log_prob(action))


def test_kl_stop_and_nonfinite_update_preserves_checkpoint(tmp_path, monkeypatch):
    from edge_sim_learning.experiment import train
    from edge_sim_learning.stability import StableExperiment

    common = dict(
        scenario=ScenarioConfig.profile("smoke"),
        episodes=2,
        workers=1,
        frames_per_batch=16,
        minibatch=16,
        epochs=3,
        eval_interval=16,
        eval_episodes=1,
    )
    monkeypatch.setattr(StableExperiment, "target_kl", 1e-12)
    train(output=tmp_path / "kl", **common)
    import csv

    rows = list(csv.DictReader((tmp_path / "kl/metrics.csv").open()))
    assert any(r["metric"] == "train/kl_early_stop" and float(r["value"]) == 1 for r in rows)
    assert any(r["metric"] == "train/actor_updates" and float(r["value"]) == 1 for r in rows)
    original = torch.optim.Adam.step

    def corrupt(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", corrupt)
    with pytest.raises(FloatingPointError, match="last healthy checkpoint"):
        train(output=tmp_path / "bad", **common)
    payload = torch.load(tmp_path / "bad/last.pt", weights_only=False)
    assert payload["experiment"]["state"]["total_frames"] == 0
    assert all(torch.isfinite(p).all() for p in payload["policy"].parameters())
    assert (tmp_path / "bad/failure.json").exists()


def test_configurable_backbone_and_initial_exploration():
    from tensordict.nn import NormalParamExtractor

    actor = HistoryActor(hidden_size=256, context_size=128, initial_std=0.3)
    features = torch.randn(8, FEATURES)
    features[:, -1] = 4
    loc, scale = NormalParamExtractor(scale_mapping="biased_softplus_1.0")(actor(features))
    assert actor.gru.hidden_size == 128
    assert actor.mlp[0].out_features == 256
    assert torch.allclose(scale, torch.full_like(scale, 0.3), atol=1e-6)
    assert loc.shape == (8, ACTION)
    (loc.sum() + scale.sum()).backward()
    assert actor.gru.weight_ih_l0.grad.abs().sum() > 0


def test_positive_reward_scaling_preserves_business_metrics():
    cfg = ScenarioConfig.profile("smoke")
    results = []
    for scale in (1.0, 0.1):
        env = SchedulingEnv(cfg.model_copy(update={"reward_scale": scale}))
        try:
            env.reset(seed=42)
            actions = {a: np.array([0, 0, 0, -5, 0.6], np.float32) for a in env.agents}
            rewards = []
            while env.agents:
                _, reward, _, _, _ = env.step(actions)
                rewards.append(next(iter(reward.values())))
            results.append((np.array(rewards), env.drain()))
        finally:
            env.close()
    np.testing.assert_allclose(results[0][0] * 0.1, results[1][0])
    assert results[0][1] == results[1][1]
