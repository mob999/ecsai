"""Check supervision semantics and actual BenchMARL transfer, including frozen resume."""

import json

import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.env import OBS
from edge_sim_learning.experiment import train
from edge_sim_learning.model import ContextModel, HistoryActor
from edge_sim_learning.pretrain import Demonstrations, collect, distribution, fit
from edge_sim_learning.scenario import ScenarioConfig


def test_pretrain_to_cross_scale_head_resume(tmp_path):
    source = ScenarioConfig.profile("smoke").model_copy(update={"scheduler_release": "window"})
    options = dict(
        method="MAPPO-no-context",
        episodes=2,
        workers=1,
        frames_per_batch=16,
        epochs=1,
        minibatch=16,
        eval_interval=16,
        eval_episodes=1,
        hidden_size=32,
        initial_std=0.3,
    )
    train(source, tmp_path / "teacher", **options)
    teacher = tmp_path / "teacher/last.pt"
    manifest = collect(
        teacher, tmp_path / "data", train_episodes=2, validation_episodes=1, workers=1
    )
    original = manifest.read_bytes()
    collect(teacher, tmp_path / "data", train_episodes=2, validation_episodes=1, workers=1)
    assert manifest.read_bytes() == original
    spec = json.loads(original)
    policy = torch.load(teacher, weights_only=False)["policy"]
    for row in spec["episodes"]:
        data = torch.load(manifest.parent / row["file"], weights_only=True)
        assert torch.equal(data["next_observation"][:-1], data["observation"][1:])
        assert not data["terminated"].any()
        assert data["truncated"].sum() == 1 and data["truncated"][-1]
        _, completed, timed_out, rejected, _, _ = data["window_counters"].T
        expected = (
            completed
            - timed_out
            - rejected
            - 0.1 * data["success_latency_sum_s"] / source.deadline_s
        ) / max(1, source.request_rate * source.period_s)
        torch.testing.assert_close(data["reward"].double(), expected, atol=1e-7, rtol=1e-6)
        assert data["metrics"]["unfinished"] == 0
        # Exact distribution match with the original BenchMARL actor on recorded features.
        actors = next(m.actors for m in policy.modules() if isinstance(m, ContextModel))
        for i, actor in enumerate(actors):
            dist, _, _ = distribution(actor, data["observation"][:, i])
            torch.testing.assert_close(dist.log_prob(data["action"][:, i]), data["log_prob"][:, i])
    base = fit([manifest], tmp_path / "base", epochs=1, batches_per_epoch=4, hidden_size=32)
    fit([manifest], tmp_path / "base", epochs=2, batches_per_epoch=4, hidden_size=32, resume=True)
    assert torch.load(tmp_path / "base/last.pt", weights_only=True)["epoch"] == 2
    initial = torch.load(base, weights_only=True)["actor"]
    target = source.model_copy(update={"clusters": 3, "caches": 3})
    train(target, tmp_path / "head", actor_init=base, head_only=True, **options)
    # Resume does not require the source base file; freezing is persisted in the checkpoint.
    train(
        target,
        tmp_path / "resumed",
        resume=tmp_path / "head/last.pt",
        **(options | {"episodes": 4}),
    )
    payload = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    actors = next(m.actors for m in payload["policy"].modules() if isinstance(m, ContextModel))
    assert len(actors) == 3
    assert len({a.mlp[-1].weight.data_ptr() for a in actors}) == 3
    for actor in actors:
        for key, value in actor.state_dict().items():
            if not key.startswith("mlp.4"):
                assert torch.equal(value, initial[key]), key
        assert not torch.equal(actor.mlp[-1].weight, initial["mlp.4.weight"])
    assert payload["experiment"]["state"]["total_frames"] == 32
    with pytest.raises(ValueError, match="checksum"):
        shard = manifest.parent / spec["episodes"][0]["file"]
        with shard.open("ab") as f:
            f.write(b"tampered")
        Demonstrations([manifest], "train")


def test_bc_gradient_and_bounds():
    torch.manual_seed(8)
    actor = HistoryActor(use_context=False, hidden_size=32, initial_std=0.3)
    x = torch.randn(64, OBS)
    actions = torch.tensor([1.0, 0.5, -0.5, 0.8, 0.55]).repeat(64, 1)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.003)
    before = -distribution(actor, x)[0].log_prob(actions).mean().item()
    for _ in range(30):
        loss = -distribution(actor, x)[0].log_prob(actions).mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
        optimizer.step()
    assert loss.item() < before
    extremes = torch.tensor([[-5, -5, -5, -5, 0.05], [5, 5, 5, 5, 0.95]])
    assert torch.isfinite(distribution(actor, x[:2])[0].log_prob(extremes)).all()
