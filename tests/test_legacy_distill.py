"""Agent identity, full epochs, learned variance, and exact distillation resume."""

import json

import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.legacy_distill import fit
from edge_sim_learning.model import HistoryActor
from edge_sim_learning.pretrain import FORMAT, SPLIT_SEEDS, distribution, sha256


def test_independent_distribution_distillation_and_resume(tmp_path):
    records = []
    for split, seed in SPLIT_SEEDS.items():
        path = tmp_path / f"{split}.pt"
        # Identical observations require opposite means for different agent identities.
        torch.save(
            dict(
                observation=torch.ones(9, 2, 13),
                action=torch.zeros(9, 2, 5),
                loc=torch.tensor([[-0.4] * 5, [0.4] * 5]).expand(9, -1, -1).clone(),
                scale=torch.full((9, 2, 5), 0.5),
            ),
            path,
        )
        records.append(dict(file=path.name, sha256=sha256(path), split=split, seed=seed))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                spec=dict(
                    format=FORMAT, scenario=dict(clusters=2), counts=dict(train=1, validation=1)
                ),
                episodes=records,
            )
        )
    )
    fit(manifest, tmp_path / "resumed", epochs=4, batch_size=4)
    fit(manifest, tmp_path / "resumed", epochs=12, batch_size=4, resume=True)
    fit(manifest, tmp_path / "direct", epochs=12, batch_size=4)
    a = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    b = torch.load(tmp_path / "direct/last.pt", weights_only=True)
    assert a["epoch"] == 12 and len(a["history"]) == 13
    assert all(row["samples"] == 18 for row in a["history"][1:])
    assert a["history"][-1]["validation_kl"] < a["history"][0]["validation_kl"]
    for left, right in zip(a["actors"], b["actors"], strict=True):
        for key in left:
            assert torch.equal(left[key], right[key])
    means = []
    for state in a["actors"]:
        actor = HistoryActor(use_context=False, hidden_size=256, initial_std=0.3)
        actor.load_state_dict(state)
        _, loc, scale = distribution(actor, torch.ones(1, 13))
        means.append(loc.mean().item())
        assert not torch.allclose(scale, torch.full_like(scale, 0.3))
    assert means[0] < 0 < means[1]
    with pytest.raises(ValueError, match="configuration mismatch"):
        fit(manifest, tmp_path / "resumed", epochs=13, batch_size=8, resume=True)


def test_mixed_loads_full_epochs_and_physical_guard(tmp_path):
    manifests = []
    for load in [0.75, 0.875, 1.0]:
        folder = tmp_path / str(load)
        folder.mkdir()
        rows = []
        for split, seed in SPLIT_SEEDS.items():
            shard = folder / (split + ".pt")
            torch.save(
                dict(
                    observation=torch.ones(4, 2, 13) * load,
                    loc=torch.zeros(4, 2, 5),
                    scale=torch.full((4, 2, 5), 0.3),
                ),
                shard,
            )
            rows.append(dict(file=shard.name, sha256=sha256(shard), split=split, seed=seed))
        manifest = folder / "manifest.json"
        manifest.write_text(
            json.dumps(
                dict(
                    spec=dict(
                        format=FORMAT,
                        scenario=dict(clusters=2, delivery_load=load, request_rate=400 * load),
                        counts=dict(train=1, validation=1),
                    ),
                    episodes=rows,
                )
            )
        )
        manifests.append(manifest)
    fit(manifests, tmp_path / "mixed", epochs=1, batch_size=5)
    fit(manifests, tmp_path / "mixed", epochs=2, batch_size=5, resume=True)
    fit(manifests, tmp_path / "direct", epochs=2, batch_size=5)
    a = torch.load(tmp_path / "mixed/last.pt", weights_only=True)
    b = torch.load(tmp_path / "direct/last.pt", weights_only=True)
    assert a["history"][-1]["samples"] == 3 * 4 * 2
    assert len(a["config"]["manifest_sha256"]) == 3
    for left, right in zip(a["actors"], b["actors"], strict=True):
        assert all(torch.equal(left[k], right[k]) for k in left)
    changed = json.loads(manifests[-1].read_text())
    changed["spec"]["scenario"]["clusters"] = 3
    manifests[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="physical scenarios"):
        fit(manifests, tmp_path / "invalid", epochs=1)
