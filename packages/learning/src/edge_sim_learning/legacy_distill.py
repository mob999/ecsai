"""Full-epoch distribution distillation without changing the legacy MAPPO actor."""

import copy
import csv
import json
from pathlib import Path

import torch
from torch import nn

from .model import ContextModel, HistoryActor
from .pretrain import Demonstrations, atomic_save, distribution, sha256

FORMAT = "edge-distill-independent-v1"


def actors_in(policy):
    return next(module.actors for module in policy.modules() if isinstance(module, ContextModel))


def distilled_policy(teacher_policy, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = copy.deepcopy(teacher_policy).cpu().eval()
    actors = actors_in(policy)
    if payload["config"]["format"] != FORMAT or len(actors) != payload["config"]["agents"]:
        raise ValueError("independent actor contract mismatch")
    for actor, state in zip(actors, payload["actors"], strict=True):
        actor.load_state_dict(state, strict=True)
    return policy


def fit(
    manifest,
    output,
    epochs=20,
    batch_size=128,
    learning_rate=3e-4,
    seed=0,
    resume=False,
    on_epoch=None,
):
    if min(epochs, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("positive distillation settings required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(manifest).read_text())
    agents = info["spec"]["scenario"]["clusters"]
    arrays = {}
    for split in ("train", "validation"):
        data = Demonstrations([manifest], split)
        shards = [data.load(path) for path in data.sources[0]]
        arrays[split] = tuple(
            torch.cat([s[key][..., :13] if key == "observation" else s[key] for s in shards])
            for key in ("observation", "loc", "scale")
        )
        if arrays[split][0].shape[1:] != (agents, 13):
            raise ValueError("demonstrations must preserve agent identity and 13 local features")
    config = dict(
        format=FORMAT,
        input_dim=13,
        action_dim=5,
        hidden_size=256,
        agents=agents,
        manifest_sha256=sha256(manifest),
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        objective="KL(teacher || student)",
        full_epochs=True,
        learnable_scale=True,
    )
    actors = nn.ModuleList(
        [HistoryActor(use_context=False, hidden_size=256, initial_std=0.3) for _ in range(agents)]
    )
    optimizer = torch.optim.Adam(actors.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    last = output / "last.pt"
    start, best, history = 0, float("inf"), []
    if resume:
        payload = torch.load(last, map_location="cpu", weights_only=True)
        if payload["config"] != config:
            raise ValueError("distillation resume configuration mismatch")
        for actor, state in zip(actors, payload["actors"], strict=True):
            actor.load_state_dict(state)
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["shuffle_rng"])
        torch.set_rng_state(payload["torch_rng"])
        start, best, history = payload["epoch"], payload["best_kl"], payload["history"]
    elif last.exists():
        raise ValueError("distillation output exists; use resume")

    def loss_at(split, indices):
        observations, loc, scale = [x[indices] for x in arrays[split]]
        terms = []
        for i, actor in enumerate(actors):
            _, student_loc, student_scale = distribution(actor, observations[:, i])
            terms.append(
                torch.distributions.kl_divergence(
                    torch.distributions.Normal(loc[:, i], scale[:, i]),
                    torch.distributions.Normal(student_loc, student_scale),
                ).sum(-1)
            )
        # The same invertible tanh/affine transform makes this the bounded-action KL too.
        return torch.stack(terms, -1)

    def validation():
        with torch.no_grad():
            return loss_at("validation", slice(None)).mean().item()

    def checkpoint(epoch):
        return dict(
            config=config,
            actors=[a.state_dict() for a in actors],
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            best_kl=best,
            history=history,
            shuffle_rng=generator.get_state(),
            torch_rng=torch.get_rng_state(),
        )

    if not resume:
        best = validation()
        history.append(dict(epoch=0, train_kl=None, validation_kl=best, samples=0))
        atomic_save(checkpoint(0), last)
        atomic_save(checkpoint(0), output / "best.pt")
        if on_epoch is not None:
            on_epoch(history[-1])
    for epoch in range(start + 1, epochs + 1):
        order = torch.randperm(len(arrays["train"][0]), generator=generator)
        total, count = 0.0, 0
        for indices in order.split(batch_size):
            loss = loss_at("train", indices).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "nonfinite distillation loss; last complete epoch retained"
                )
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(actors.parameters(), 0.5)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite distillation gradient")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in actors.parameters()):
                raise FloatingPointError("nonfinite student parameters")
            total += loss.item() * len(indices)
            count += len(indices)
        score = validation()
        if not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("nonfinite validation KL")
        history.append(
            dict(epoch=epoch, train_kl=total / count, validation_kl=score, samples=count * agents)
        )
        improved = score < best
        best = min(best, score)
        atomic_save(checkpoint(epoch), last)
        if improved:
            atomic_save(checkpoint(epoch), output / "best.pt")
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if on_epoch is not None:
            on_epoch(history[-1])
    return output / "best.pt"
