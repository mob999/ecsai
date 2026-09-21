"""Episode demonstrations and a portable local actor; no simulator dependencies added."""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal

from .env import ACTION, OBS, SchedulingEnv
from .model import HistoryActor
from .scenario import ScenarioConfig

FORMAT = "edge-bc-v1"
COUNTERS = ("arrived", "completed", "timed_out", "rejected", "backhaul_bytes", "delivery_bytes")
SPLIT_SEEDS = {"train": 100_000_000, "validation": 200_000_000}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def distribution(actor, observation):
    loc, scale = NormalParamExtractor()(actor(observation))
    low = observation.new_tensor([-5, -5, -5, -5, 0.05])
    high = observation.new_tensor([5, 5, 5, 5, 0.95])
    return TanhNormal(loc, scale, low=low, high=high), loc, scale


def load_base_policy(path):
    """Same local student for any agent count, with the BenchMARL action transform."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    if config["format"] == "edge-bc-v2":
        from .multiscale_bc import policy_from_payload

        return policy_from_payload(payload)
    if config["format"] != FORMAT or config["input_dim"] != OBS or config["action_dim"] != ACTION:
        raise ValueError("unknown base actor contract")
    actor = HistoryActor(use_context=False, hidden_size=config["hidden_size"])
    actor.load_state_dict(payload["actor"])
    module = TensorDictSequential(
        TensorDictModule(actor, in_keys=[("agents", "observation")], out_keys=[("agents", "raw")]),
        TensorDictModule(
            NormalParamExtractor(),
            in_keys=[("agents", "raw")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        ),
    )
    return ProbabilisticActor(
        module=module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.tensor([-5, -5, -5, -5, 0.05]),
            "high": torch.tensor([5, 5, 5, 5, 0.95]),
        },
        return_log_prob=True,
        log_prob_key=("agents", "log_prob"),
    )


def policy_input(observations, agents):
    x = torch.from_numpy(np.stack([observations[a] for a in agents]))
    return TensorDict(
        {"agents": TensorDict({"observation": x}, [len(agents)]), "state": x[:, :OBS].reshape(-1)},
        [],
    )


def _collect_episode(job):
    teacher, folder, split, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    payload = torch.load(teacher, map_location="cpu", weights_only=False)
    if payload["method"] != "MAPPO-no-context" or payload.get("action_dim") != ACTION:
        raise ValueError("BC teacher must be a five-action MAPPO-no-context checkpoint")
    config = ScenarioConfig.model_validate(payload["scenario"])
    policy = payload["policy"].cpu().eval()
    env = SchedulingEnv(config, evaluation=True, method="MAPPO-no-context")
    columns = {
        k: []
        for k in (
            "observation",
            "action",
            "loc",
            "scale",
            "log_prob",
            "next_observation",
            "reward",
            "terminated",
            "truncated",
            "window_counters",
            "success_latency_sum_s",
        )
    }
    started = perf_counter()
    try:
        observations, _ = env.reset(seed=seed)

        def infer(obs):
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                td = policy(policy_input(obs, env.possible_agents))
            return td, dict(zip(env.possible_agents, td["agents", "action"].numpy(), strict=True))

        while env.agents:
            td, actions = infer(observations)
            before = env.metrics()
            latency_count = len(env.latencies)
            following, rewards, terminated, truncated, _ = env.step(actions)
            for key in ("observation", "action", "loc", "scale", "log_prob"):
                columns[key].append(td["agents", key].detach().clone())
            columns["next_observation"].append(
                policy_input(following, env.possible_agents)["agents", "observation"]
            )
            columns["reward"].append(torch.tensor(next(iter(rewards.values()))))
            columns["terminated"].append(torch.tensor(all(terminated.values())))
            columns["truncated"].append(torch.tensor(all(truncated.values())))
            after = env.metrics()
            columns["window_counters"].append(
                torch.tensor([after[k] - before[k] for k in COUNTERS], dtype=torch.float64)
            )
            columns["success_latency_sum_s"].append(
                torch.tensor(sum(env.latencies[latency_count:]), dtype=torch.float64)
            )
            observations = following
        at_truncation = env.metrics()
        final = env.drain(lambda obs: infer(obs)[1])
        if final["unfinished"] != 0 or final["arrived"] != sum(
            final[k] for k in ("completed", "timed_out", "rejected")
        ):
            raise ValueError("demonstration drain did not reconcile request counts")
        data = {k: torch.stack(v) for k, v in columns.items()}
        data.update(
            format=FORMAT,
            seed=seed,
            split=split,
            workload=env.workload,
            scenario=config.model_dump(),
            metrics=final,
            truncated_metrics=at_truncation,
            episode_return=env.episode_return,
            wall_s=perf_counter() - started,
            counter_names=list(COUNTERS),
        )
        path = Path(folder) / f"{split}-{seed}.pt"
        atomic_save(data, path)
        return {
            "file": path.name,
            "sha256": sha256(path),
            "split": split,
            "seed": seed,
            "steps": config.cycles,
            "agents": config.clusters,
            "wall_s": data["wall_s"],
        }
    finally:
        env.close()


def collect(teacher, output, train_episodes=128, validation_episodes=32, workers=2):
    teacher, output = Path(teacher).resolve(), Path(output).resolve()
    if min(train_episodes, validation_episodes, workers) < 1:
        raise ValueError("episode and worker counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    source = torch.load(teacher, map_location="cpu", weights_only=False)
    spec = {
        "format": FORMAT,
        "teacher_sha256": sha256(teacher),
        "scenario": json.loads(json.dumps(source["scenario"])),
        "teacher_frames": source["experiment"]["state"]["total_frames"],
        "counts": {"train": train_episodes, "validation": validation_episodes},
        "seed_bases": SPLIT_SEEDS,
    }
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous["spec"] != spec:
        raise ValueError("demonstration output belongs to another collection specification")
    records = previous.get("episodes", [])
    for row in records:
        if sha256(output / row["file"]) != row["sha256"]:
            raise ValueError("demonstration checksum mismatch")
    done = {(row["split"], row["seed"]) for row in records}
    jobs = [
        (str(teacher), str(output), split, SPLIT_SEEDS[split] + i)
        for split, count in spec["counts"].items()
        for i in range(count)
        if (split, SPLIT_SEEDS[split] + i) not in done
    ]
    started = perf_counter()

    def record(row):
        records.append(row)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "spec": spec,
                    "episodes": records,
                    "collection_wall_s": previous.get("collection_wall_s", 0)
                    + perf_counter()
                    - started,
                },
                indent=2,
            )
        )
        temporary.replace(manifest_path)

    if workers == 1:
        for job in jobs:
            record(_collect_episode(job))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for row in pool.map(_collect_episode, jobs):
                record(row)
    return manifest_path


class Demonstrations:
    """Whole-episode splits, equal source weights, bounded on-disk shard loading."""

    def __init__(self, manifests, split):
        self.sources = []
        self.hashes = []
        seen = set()
        for path in map(Path, manifests):
            manifest = json.loads(path.read_text())
            if manifest["spec"]["format"] != FORMAT:
                raise ValueError("unknown demonstration format")
            for candidate_split, count in manifest["spec"]["counts"].items():
                expected = set(
                    range(SPLIT_SEEDS[candidate_split], SPLIT_SEEDS[candidate_split] + count)
                )
                actual = [r["seed"] for r in manifest["episodes"] if r["split"] == candidate_split]
                if set(actual) != expected or len(actual) != count:
                    raise ValueError("incomplete or overlapping episode seed split")
            source = []
            for row in manifest["episodes"]:
                if row["split"] != split:
                    continue
                identity = (json.dumps(manifest["spec"]["scenario"], sort_keys=True), row["seed"])
                if identity in seen:
                    raise ValueError("duplicate workload in demonstration sources")
                seen.add(identity)
                shard = path.parent / row["file"]
                if sha256(shard) != row["sha256"]:
                    raise ValueError("demonstration checksum mismatch")
                source.append(shard)
            if len(source) != manifest["spec"]["counts"][split]:
                raise ValueError("incomplete demonstration split")
            self.sources.append(source)
            self.hashes.append(sha256(path))
        if not self.sources:
            raise ValueError("at least one demonstration source is required")
        self._cache = {}

    def load(self, path):
        if path not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = torch.load(path, map_location="cpu", weights_only=True)
        return self._cache[path]

    def sample(self, count, generator):
        # One uniformly selected source/episode per batch; samples vary across steps/agents.
        source = self.sources[int(torch.randint(len(self.sources), (), generator=generator))]
        data = self.load(source[int(torch.randint(len(source), (), generator=generator))])
        n = data["action"].shape[0] * data["action"].shape[1]
        indices = torch.randint(n, (count,), generator=generator)
        return tuple(
            data[k].reshape(n, -1)[indices] for k in ("observation", "action", "loc", "scale")
        )

    def validation(self, actor, device, batch_size):
        source_scores = []
        with torch.no_grad():
            for source in self.sources:
                total, count = np.zeros(4), 0
                for path in source:
                    data = self.load(path)
                    n = data["action"].numel() // ACTION
                    arrays = [
                        data[k].reshape(n, -1) for k in ("observation", "action", "loc", "scale")
                    ]
                    for start in range(0, n, batch_size):
                        obs, action, loc, scale = [
                            x[start : start + batch_size].to(device) for x in arrays
                        ]
                        dist, student_loc, student_scale = distribution(actor, obs)
                        kl = torch.distributions.kl_divergence(
                            torch.distributions.Normal(loc, scale),
                            torch.distributions.Normal(student_loc, student_scale),
                        ).sum(-1)
                        low = action.new_tensor([-5, -5, -5, -5, 0.05])
                        high = action.new_tensor([5, 5, 5, 5, 0.95])
                        action_fraction = (action - low) / (high - low)
                        mean_fraction = (dist.deterministic_sample - low) / (high - low)
                        total += [
                            -dist.log_prob(action).sum().item(),
                            kl.sum().item(),
                            ((action_fraction < 0.01) | (action_fraction > 0.99))
                            .float()
                            .mean(-1)
                            .sum()
                            .item(),
                            ((mean_fraction < 0.01) | (mean_fraction > 0.99))
                            .float()
                            .mean(-1)
                            .sum()
                            .item(),
                        ]
                        count += len(obs)
                source_scores.append(total / count)
        return np.mean(source_scores, axis=0)


def fit(
    manifests,
    output,
    epochs=20,
    batches_per_epoch=128,
    batch_size=256,
    hidden_size=256,
    learning_rate=3e-4,
    seed=0,
    device="cpu",
    mode="offline",
    resume=False,
):
    import wandb

    if min(epochs, batches_per_epoch, batch_size, hidden_size) < 1 or learning_rate <= 0:
        raise ValueError("BC training counts and learning rate must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_data, validation_data = (
        Demonstrations(manifests, split) for split in ("train", "validation")
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise ValueError("BC output already contains a run; use a new output directory")
    actor = HistoryActor(use_context=False, hidden_size=hidden_size, initial_std=0.3).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    config = dict(
        format=FORMAT,
        hidden_size=hidden_size,
        input_dim=OBS,
        action_dim=ACTION,
        manifests=train_data.hashes,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        learning_rate=learning_rate,
    )
    first_epoch, previous_wall, best = 0, 0.0, math.inf
    if resume:
        payload = torch.load(output / "last.pt", map_location=device, weights_only=True)
        previous = payload["config"] | {"epochs": epochs}
        if previous != config or payload["epoch"] >= epochs:
            raise ValueError("BC resume must preserve configuration and extend the epoch budget")
        actor.load_state_dict(payload["actor"])
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["sample_rng"].cpu())
        torch.set_rng_state(payload["torch_rng"].cpu())
        first_epoch, previous_wall = payload["epoch"] + 1, payload["metrics"]["wall_s"]
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)["metrics"][
            "validation_nll"
        ]
    (output / "config.json").write_text(json.dumps(config, indent=2))
    run = wandb.init(
        project="ecsai-deppo",
        name=os.environ.get("ECSAI_RUN_NAME", "BC-base"),
        mode=mode,
        dir=str(output),
        config=config,
    )
    started = perf_counter()
    try:
        with (output / "metrics.csv").open("a" if resume else "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "updates",
                    "wall_s",
                    "train_nll",
                    "validation_nll",
                    "teacher_kl",
                    "teacher_action_saturation",
                    "student_mean_saturation",
                ],
            )
            if not resume:
                writer.writeheader()
            for epoch in range(first_epoch, epochs + 1):
                losses = []
                if epoch:
                    actor.train()
                    for _ in range(batches_per_epoch):
                        obs, actions, _, _ = [
                            x.to(device) for x in train_data.sample(batch_size, generator)
                        ]
                        dist, _, _ = distribution(actor, obs)
                        loss = -dist.log_prob(actions).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "non-finite BC loss; last.pt preserves previous epoch"
                            )
                        optimizer.zero_grad()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("non-finite BC gradient")
                        optimizer.step()
                        if not all(torch.isfinite(p).all() for p in actor.parameters()):
                            raise FloatingPointError("non-finite BC parameter")
                        losses.append(loss.item())
                actor.eval()
                nll, kl, action_sat, mean_sat = validation_data.validation(
                    actor, device, batch_size
                )
                if not np.isfinite([nll, kl, action_sat, mean_sat]).all():
                    raise FloatingPointError("non-finite BC validation")
                row = dict(
                    epoch=epoch,
                    updates=epoch * batches_per_epoch,
                    wall_s=previous_wall + perf_counter() - started,
                    train_nll=float(np.mean(losses)) if losses else float(nll),
                    validation_nll=float(nll),
                    teacher_kl=float(kl),
                    teacher_action_saturation=float(action_sat),
                    student_mean_saturation=float(mean_sat),
                )
                writer.writerow(row)
                f.flush()
                run.log(row)
                payload = dict(
                    config=config,
                    actor={k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    metrics=row,
                    torch_rng=torch.get_rng_state(),
                    sample_rng=generator.get_state(),
                )
                atomic_save(payload, output / "last.pt")
                if nll < best:
                    best = nll
                    atomic_save(payload, output / "best.pt")
    finally:
        run.finish()
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-episodes", type=int, default=128)
    p.add_argument("--validation-episodes", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("fit")
    p.add_argument("--manifests", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batches-per-epoch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--mode", choices=["offline", "online"], default="offline")
    p.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(collect(**args) if command == "collect" else fit(**args))


if __name__ == "__main__":
    main()
