"""Cross-scale contextual MAPPO transfer and paired load schedules."""

import json

import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.experiment import train
from edge_sim_learning.local_context import PROFILE
from edge_sim_learning.model import ContextModel
from edge_sim_learning.multiscale_bc import FORMAT, ContextActor
from edge_sim_learning.scenario import ScenarioConfig
from edge_sim_learning.torch_env import ContentBatchEnv


@pytest.mark.parametrize("loads", [(0.25, 0.5, 0.75, 1, 1.25), (0.75,)])
def test_context_load_cycle_and_state(loads):
    cfg = ScenarioConfig.profile("smoke").model_copy(update={"cycles": 10, "max_retries": 2})
    env = ContentBatchEnv(
        cfg,
        workers=1,
        method="MAPPO-no-context",
        local_context=True,
        load_mix=loads,
    )
    try:
        seen = []
        capacities = []
        for _ in range(2):
            td = env.reset()
            seen = []
            session = env.envs[0].session
            capacities.append([c.total_bandwidth_bytes_s for c in env.envs[0].run.content.caches])
            assert td["agents", "observation"].shape[-1] == 21
            assert td["state"].shape[-1] == 21 * cfg.clusters
            for _ in range(10):
                td = env.step(env.rand_action(td))["next"]
                seen.append(round(td["metrics", "delivery_load"].item(), 2))
                assert env.envs[0].session is session
            assert seen == [load for load in loads for _ in range(10 // len(loads))]
        assert all(x == capacities[0] for x in capacities)
    finally:
        env.close()


@pytest.mark.parametrize("reward_mode", ["business", "logical"])
def test_context_pretrained_vs_scratch_update_and_resume(tmp_path, reward_mode):
    torch.manual_seed(12)
    actor = ContextActor(256, 4, True, 0.05, 0.1)
    config = dict(
        format=FORMAT,
        observation_profile=PROFILE,
        input_dim=21,
        action_dim=5,
        hidden_size=256,
        architecture=dict(depth=4, layer_norm=True, dropout=0.05),
        fixed_scale=0.1,
    )
    base = tmp_path / "base.pt"
    torch.save(dict(config=config, actor=actor.state_dict()), base)
    cfg = ScenarioConfig.profile("smoke").model_copy(
        update={"cycles": 4, "max_retries": 2, "reward_mode": reward_mode}
    )
    opts = dict(
        method="MAPPO-no-context",
        workers=1,
        frames_per_batch=8,
        epochs=1,
        minibatch=8,
        eval_interval=8,
        eval_episodes=1,
        hidden_size=256,
        local_context=True,
        load_mix=(0.25, 1.25),
    )
    for arm in ["pretrained", "scratch"]:
        train(
            cfg,
            tmp_path / arm,
            episodes=2,
            actor_init=base if arm == "pretrained" else None,
            **opts,
        )
    initial = torch.load(tmp_path / "pretrained/initial.pt", weights_only=False)
    last = torch.load(tmp_path / "pretrained/last.pt", weights_only=False)
    scratch = torch.load(tmp_path / "scratch/initial.pt", weights_only=False)
    critic_keys = [k for k in initial["experiment"]["loss_agents"] if "critic" in k]
    assert critic_keys
    for key in critic_keys:
        a = initial["experiment"]["loss_agents"][key]
        b = scratch["experiment"]["loss_agents"][key]
        if torch.is_tensor(a):
            assert torch.equal(a, b), key

    def actors(p):
        return next(m.actors for m in p["policy"].modules() if isinstance(m, ContextModel))

    for a in actors(initial):
        for k, v in a.state_dict().items():
            assert torch.equal(v, actor.state_dict()[k])
        assert not any(isinstance(m, torch.nn.Dropout) for m in a.modules())
        actor.eval()
        features = torch.randn(4, 21)
        torch.testing.assert_close(a(features), actor(features))
    assert len({a.mlp[0].weight.data_ptr() for a in actors(initial)}) == cfg.clusters
    assert not torch.equal(actors(initial)[0].mlp[0].weight, actors(scratch)[0].mlp[0].weight)
    assert any(
        not torch.equal(v, actors(initial)[0].state_dict()[k])
        for k, v in actors(last)[0].state_dict().items()
    )
    report = json.loads((tmp_path / "pretrained/evaluation-context-8.json").read_text())
    assert set(report["conditions"]) == {"within-episode"}
    train(cfg, tmp_path / "resumed", episodes=4, resume=tmp_path / "pretrained/last.pt", **opts)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert resumed["experiment"]["state"]["total_frames"] == 16


def test_piecewise_poisson_rates_and_capacity_are_reproducible():
    from edge_sim_learning.scenario import build_run

    base = ScenarioConfig.profile("smoke").model_copy(update={"cycles": 50, "request_rate": 1000})
    cfg = base.model_copy(update={"episode_loads": (0.25, 0.5, 0.75, 1, 1.25)})
    dynamic, metadata = build_run(cfg, 123)
    repeated, same = build_run(cfg, 123)
    static, _ = build_run(base, 123)
    assert dynamic.content.requests == repeated.content.requests
    assert metadata == same
    assert [c.total_bandwidth_bytes_s for c in dynamic.content.caches] == [
        c.total_bandwidth_bytes_s for c in static.content.caches
    ]
    assert len(metadata["load_phases"]) == 5
    for start, end, load, rate in cfg.load_phases():
        count = sum(start <= r.arrival_s < end for r in dynamic.content.requests)
        assert abs(count - rate * (end - start)) < 0.25 * rate * (end - start)
        assert rate == pytest.approx(base.request_rate * load / base.delivery_load)
        assert cfg.expected_arrivals(start, end) == pytest.approx(rate * (end - start))
    assert cfg.expected_arrivals(0.9, 1.1) == pytest.approx(100)
