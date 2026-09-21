"""Balanced, resumable multi-scenario BC. No RL training is invoked here."""

import argparse
import csv
import gzip
import json
import math
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal

from .experiment import evaluate, metadata
from .local_context import CONTEXT_DIM, CONTEXT_NAMES, PROFILE, LocalContextEnv
from .pretrain import COUNTERS, atomic_save, distribution, sha256
from .scenario import ScenarioConfig, build_run

FORMAT = "edge-bc-v2"
TEACHERS = [
    f"{side}-{ratio}"
    for side in ("local", "forward")
    for ratio in ("0.4", "0.5", "0.6", "adaptive")
]
BASELINES = {
    "Random": ("random", None),
    "Always-local": ("local", None),
    "Always-forward": ("forward", None),
    "Forward-r0.6": ("forward", 0.6),
    "Queue-adaptive": ("queue-adaptive", None),
}
# Nonoverlapping namespaces, also distinct from the old v1 data/evaluation streams.
SEED_BASES = {
    "selection": 3_410_000_000,
    "train": 3_420_000_000,
    "supervised-validation": 3_430_000_000,
    "validation": 3_440_000_000,
    "test": 3_450_000_000,
    "dagger": 3_460_000_000,
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def freeze_spec(path, spec):
    spec = json.loads(json.dumps(spec))
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError(f"changed specification: {path}")
    if not path.exists():
        write_json(path, spec)


def seeds(split, condition_index, count, round_index=0):
    first = SEED_BASES[split] + condition_index * 10_000 + round_index * 1000
    if count >= 1000:
        raise ValueError("seed allocation exceeded")
    return list(range(first, first + count))


class ContextActor(nn.Module):
    """A 21-feature actor; deliberately separate from HistoryActor's DD mask."""

    def __init__(self, hidden_size=256, depth=2, layer_norm=False, dropout=0.0, fixed_scale=None):
        super().__init__()
        self.fixed_scale = fixed_scale
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(CONTEXT_DIM if i == 0 else hidden_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers, nn.Linear(hidden_size, 10))
        raw = math.log(math.expm1(0.3 - 0.01)) - math.log(math.expm1(0.99))
        with torch.no_grad():
            self.mlp[-1].weight[5:].zero_()
            self.mlp[-1].bias[5:].fill_(raw)

    def forward(self, observation):
        if observation.shape[-1] != CONTEXT_DIM:
            raise ValueError("local-context-v2 requires exactly 21 features")
        loc, scale = self.mlp(observation).chunk(2, -1)
        if self.fixed_scale is not None:
            raw = math.log(math.expm1(self.fixed_scale - 0.01)) - math.log(math.expm1(0.99))
            scale = torch.full_like(scale, raw)
        return torch.cat((3 * torch.tanh(loc / 3), scale.clamp(-3, 1)), -1)


def actor_from_config(config):
    return ContextActor(
        config["hidden_size"],
        **config.get("architecture", {}),
        fixed_scale=config.get("fixed_scale"),
    )


def policy_from_payload(payload):
    config = payload["config"]
    if (
        config["format"] != FORMAT
        or config["observation_profile"] != PROFILE
        or config["input_dim"] != CONTEXT_DIM
        or config["action_dim"] != 5
    ):
        raise ValueError("incompatible v2 actor contract")
    actor = actor_from_config(config)
    actor.eval()
    actor.load_state_dict(payload["actor"])
    module = TensorDictSequential(
        TensorDictModule(actor, in_keys=[("agents", "observation")], out_keys=[("agents", "raw")]),
        TensorDictModule(
            NormalParamExtractor(),
            in_keys=[("agents", "raw")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        ),
    )
    policy = ProbabilisticActor(
        module,
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
    policy.observation_profile = PROFILE
    return policy


def teacher_method(name):
    side, ratio = name.split("-", 1)
    if name not in TEACHERS:
        raise ValueError("unknown executable teacher")
    return (f"{side}-adaptive", None) if ratio == "adaptive" else (side, float(ratio))


def teacher_actions(env, name):
    side, ratio = name.split("-", 1)
    teacher_method(name)
    result = {}
    for agent in env.possible_agents:
        r = float(ratio) if ratio != "adaptive" else 0.5
        if ratio == "adaptive":
            pools = [p for p in env.last_view.pools if env.cache_clusters[p.node_id] == agent]
            back = sum(p.remaining_bytes for p in pools if p.kind == "backhaul")
            total = sum(p.remaining_bytes for p in pools)
            r = float(np.clip(back / total, 0.05, 0.95)) if total else 0.5
        result[agent] = np.array([0, 0, 0, -2 if side == "local" else 2, r], np.float32)
    return result


def safe_actions(actions):
    low = actions.new_tensor([-5, -5, -5, -5, 0.05])
    high = actions.new_tensor([5, 5, 5, 5, 0.95])
    eps = (high - low) * 1e-5
    return torch.maximum(torch.minimum(actions, high - eps), low + eps)


def archive_audit(path, episodes):
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt") as stream:
        for ep in episodes:
            if "retry_outcomes" in ep:
                stream.write(
                    json.dumps({"seed": ep["seed"], "requests": ep.pop("retry_outcomes")}) + "\n"
                )
    temp.replace(path)


def cached_evaluate(folder, cfg, seed_list, method="local", ratio=None, checkpoint=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(
        scenario=cfg.model_dump(),
        seeds=seed_list,
        method=method,
        ratio=ratio,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        exploration="stochastic",
    )
    freeze_spec(folder / "spec.json", spec)
    dest = folder / "evaluation.json"
    if dest.exists():
        return json.loads(dest.read_text())
    print(f"EVAL {folder}", flush=True)
    policy = (
        policy_from_payload(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if checkpoint
        else None
    )
    result = evaluate(
        cfg,
        method,
        policy,
        seed_list,
        fixed_ratio=ratio,
        exploration="stochastic",
        workers=EVAL_WORKERS,
    )
    for ep in result["episodes"]:
        if ep["logical_unfinished"] or ep["unfinished"]:
            raise ValueError("incomplete retry drain")
        if ep["logical_completed"] + ep["logical_failed"] != ep["logical_requests"]:
            raise ValueError("logical request accounting mismatch")
    archive_audit(folder / "attempts.jsonl.gz", result["episodes"])
    write_json(dest, result)
    return result


EVAL_WORKERS = 1


@contextmanager
def evaluation_slot():
    """Queue independent GPU training runs behind one CPU evaluation slot."""
    path = os.environ.get("BC_EVAL_LOCK")
    if not path:
        yield
        return
    import fcntl

    with open(path, "a") as lock:
        print("EVAL_QUEUE waiting", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            print("EVAL_QUEUE acquired", flush=True)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _evaluation_job(job):
    global EVAL_WORKERS
    folder, config, seed_list, method, ratio, checkpoint, workers = job
    EVAL_WORKERS = workers
    return cached_evaluate(
        folder, ScenarioConfig.model_validate(config), seed_list, method, ratio, checkpoint
    )


def evaluation_batch(jobs):
    """Bound total simulation workers; independent condition controllers use spawn."""
    budget = EVAL_WORKERS
    parallel = min(len(jobs), max(1, budget // 4))
    inner = max(1, budget // parallel)
    tasks = [(*job, inner) for job in jobs]
    if parallel == 1:
        return [_evaluation_job(job) for job in tasks]
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_evaluation_job, tasks))


def reference_report(reports):
    # Stable ID tie-break, then success, then latency (never select on student output).
    name = min(
        reports,
        key=lambda n: (
            -reports[n]["mean"]["logical_success_rate"],
            reports[n]["mean"]["mean_success_e2e_s"],
            n,
        ),
    )
    return name, reports[name]


def gate(student, reference):
    s, b = student["mean"], reference["mean"]
    ds = s["logical_success_rate"] - b["logical_success_rate"]
    ratio = s["mean_success_e2e_s"] / max(b["mean_success_e2e_s"], 1e-12)
    deficit = max((-ds - 0.01) / 0.01, (ratio - 1.05) / 0.05)
    return dict(
        passed=ds >= -0.01 - 1e-12 and ratio <= 1.05 + 1e-12,
        success_difference_pp=ds * 100,
        latency_ratio=ratio,
        deficit=deficit,
    )


def paired_ci(student, reference):
    a = {e["seed"]: e for e in student["episodes"]}
    b = {e["seed"]: e for e in reference["episodes"]}
    if a.keys() != b.keys():
        raise ValueError("unpaired evaluation seeds")
    order = sorted(a)
    for seed in order:
        if a[seed]["workload"] != b[seed]["workload"]:
            raise ValueError("different paired workload")
    rng = np.random.default_rng(20260921)
    indices = rng.integers(len(order), size=(10000, len(order)))
    delta = np.array([a[k]["logical_success_rate"] - b[k]["logical_success_rate"] for k in order])
    sa = np.array([a[k]["mean_success_e2e_s"] for k in order])
    sb = np.array([b[k]["mean_success_e2e_s"] for k in order])
    return dict(
        success_difference_pp_ci95=(
            np.quantile(delta[indices].mean(1), [0.025, 0.975]) * 100
        ).tolist(),
        latency_ratio_ci95=np.quantile(
            sa[indices].mean(1) / np.maximum(sb[indices].mean(1), 1e-12), [0.025, 0.975]
        ).tolist(),
    )


def _collect_one(job):
    cfg_dict, teacher, path, seed, split, condition, behavior = job
    path = Path(path)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = ScenarioConfig.model_validate(cfg_dict)
    env = LocalContextEnv(cfg, evaluation=True, method="MAPPO-no-context")
    actor = None
    if behavior:
        payload = torch.load(behavior, map_location="cpu", weights_only=True)
        actor = actor_from_config(payload["config"])
        actor.load_state_dict(payload["actor"])
        actor.eval()
    columns = {
        k: []
        for k in [
            "observation",
            "action",
            "teacher_action",
            "next_observation",
            "reward",
            "terminated",
            "truncated",
            "window_counters",
        ]
    }
    started = perf_counter()
    try:
        obs, _ = env.reset(seed=seed)

        def behavior_actions(observations):
            if actor is None:
                return teacher_actions(env, teacher)
            with torch.no_grad():
                x = torch.tensor(np.stack([observations[a] for a in env.possible_agents]))
                action = distribution(actor, x)[0].sample().numpy()
            return dict(zip(env.possible_agents, action, strict=True))

        while env.agents:
            target = teacher_actions(env, teacher)
            actions = behavior_actions(obs)
            before = env.metrics()
            following, reward, terminated, truncated, _ = env.step(actions)
            for key, values in [
                ("observation", obs),
                ("action", actions),
                ("teacher_action", target),
                ("next_observation", following),
            ]:
                columns[key].append(
                    torch.tensor(np.stack([values[a] for a in env.possible_agents]))
                )
            columns["reward"].append(torch.tensor(next(iter(reward.values()))))
            columns["terminated"].append(torch.tensor(all(terminated.values())))
            columns["truncated"].append(torch.tensor(all(truncated.values())))
            after = env.metrics()
            columns["window_counters"].append(
                torch.tensor([after[k] - before[k] for k in COUNTERS], dtype=torch.float64)
            )
            obs = following
        final = env.drain(behavior_actions)
        data = {k: torch.stack(v) for k, v in columns.items()}
        data.update(
            format=FORMAT,
            observation_profile=PROFILE,
            seed=seed,
            split=split,
            condition=condition,
            scenario=cfg_dict,
            teacher=teacher,
            behavior_sha256=sha256(behavior) if behavior else None,
            workload=env.workload,
            metrics=final,
            wall_s=perf_counter() - started,
            counter_names=list(COUNTERS),
        )
        archive_audit(
            path.with_suffix(".jsonl.gz"), [{"seed": seed, "retry_outcomes": env.retry_records()}]
        )
        atomic_save(data, path)
        return dict(
            file=str(path),
            sha256=sha256(path),
            condition=condition,
            split=split,
            seed=seed,
            teacher=teacher,
        )
    finally:
        env.close()


def collect_jobs(output, jobs, workers):
    output.mkdir(parents=True, exist_ok=True)
    spec = [
        {
            "scenario": j[0],
            "teacher": j[1],
            "file": j[2],
            "seed": j[3],
            "split": j[4],
            "condition": j[5],
            "behavior_sha256": sha256(j[6]) if j[6] else None,
        }
        for j in jobs
    ]
    freeze_spec(output / "spec.json", spec)
    manifest = output / "manifest.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    for row in records:
        if sha256(row["file"]) != row["sha256"]:
            raise ValueError("dataset checksum mismatch")
    done = {r["file"] for r in records}
    pending = [j for j in jobs if j[2] not in done]
    if workers == 1:
        iterator = map(_collect_one, pending)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        iterator = pool.map(_collect_one, pending)
    try:
        for row in iterator:
            records.append(row)
            write_json(manifest, records)
            print(f"COLLECT {row['condition']}/{row['split']}/{row['seed']}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    return records


class BalancedData:
    def __init__(self, records, split):
        self.groups, self.cache = {}, {}
        identities = set()
        for r in records:
            identity = (r["condition"], r["seed"])
            if identity in identities:
                raise ValueError("duplicate episode identity")
            identities.add(identity)
            if sha256(r["file"]) != r["sha256"]:
                raise ValueError("dataset checksum mismatch")
            if r["split"] == split:
                self.groups.setdefault(r["condition"], []).append(r["file"])
        self.names = sorted(self.groups)
        if not self.names:
            raise ValueError("empty dataset split")

    def load(self, path):
        if path not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data["format"] != FORMAT or data["observation"].shape[-1] != CONTEXT_DIM:
                raise ValueError("incompatible dataset")
            self.cache[path] = data
        return self.cache[path]

    def sample(self, count, generator):
        name = self.names[int(torch.randint(len(self.names), (), generator=generator))]
        paths = self.groups[name]
        data = self.load(paths[int(torch.randint(len(paths), (), generator=generator))])
        obs = data["observation"].reshape(-1, CONTEXT_DIM)
        target = data["teacher_action"].reshape(-1, 5)
        indices = torch.randint(len(obs), (count,), generator=generator)
        return obs[indices], target[indices]

    def validation_nll(self, actor, device):
        scores = []
        with torch.no_grad():
            for name in self.names:
                total = count = 0
                for path in self.groups[name]:
                    data = self.load(path)
                    obs = data["observation"].reshape(-1, CONTEXT_DIM)
                    target = safe_actions(data["teacher_action"].reshape(-1, 5))
                    for i in range(0, len(obs), 1024):
                        loss = -distribution(actor, obs[i : i + 1024].to(device))[0].log_prob(
                            target[i : i + 1024].to(device)
                        )
                        total += loss.sum().item()
                        count += len(loss)
                scores.append(total / count)
        return float(np.mean(scores))

    def full_data(self):
        """Visit every sample once; retain equal-condition weighting in the loss."""
        observations, targets, weights = [], [], []
        for name in self.names:
            obs, labels = [], []
            for path in self.groups[name]:
                data = self.load(path)
                obs.append(data["observation"].reshape(-1, CONTEXT_DIM))
                labels.append(data["teacher_action"].reshape(-1, 5))
            obs, labels = torch.cat(obs), torch.cat(labels)
            observations.append(obs)
            targets.append(labels)
            weights.append(torch.full((len(obs),), 1 / (len(self.names) * len(obs))))
        observations = torch.cat(observations)
        return observations, torch.cat(targets), torch.cat(weights) * len(observations)


def fit_phase(
    folder,
    records,
    initial,
    device,
    hidden,
    epochs,
    batches,
    batch_size,
    interval,
    evaluate_checkpoint,
    continue_initial=False,
    architecture=None,
    weight_decay=0.0,
    full_epoch=False,
    learning_rate=3e-4,
    loss_kind="nll",
    fixed_scale=None,
):
    import wandb

    folder.mkdir(parents=True, exist_ok=True)
    config = dict(
        format=FORMAT,
        observation_profile=PROFILE,
        input_dim=CONTEXT_DIM,
        context_names=list(CONTEXT_NAMES),
        action_dim=5,
        hidden_size=hidden,
        learning_rate=learning_rate,
        batch_size=batch_size,
        batches=batches,
        epochs=epochs,
        seed=0,
        data=[r["sha256"] for r in records],
        initial_sha256=sha256(initial) if initial else None,
    )
    if architecture is not None:
        config["architecture"] = architecture
        config["weight_decay"] = weight_decay
    if full_epoch:
        config["sampling"] = "full-shuffle-condition-weighted"
    if loss_kind not in ("nll", "mean-mse"):
        raise ValueError("unknown loss")
    if loss_kind != "nll":
        if fixed_scale is None:
            raise ValueError("mean-mse requires fixed action scale")
        config["loss_kind"] = loss_kind
    if fixed_scale is not None:
        if not 0.1 <= fixed_scale <= 0.3:
            raise ValueError("fixed_scale must be between 0.1 and 0.3")
        config["fixed_scale"] = fixed_scale
    if continue_initial:
        if initial is None:
            raise ValueError("continuation requires an initial checkpoint")
        config["continue_initial"] = True
        original = torch.load(initial, map_location="cpu", weights_only=True)
        for key in ("fixed_scale", "loss_kind"):
            if original["config"].get(key) != config.get(key):
                raise ValueError(f"continuation changed {key}")
        if original["config"].get("sampling") != config.get("sampling"):
            raise ValueError("continuation changed sampling")
        if original["config"].get("architecture") != config.get("architecture"):
            raise ValueError("continuation changed architecture")
        if original["config"].get("weight_decay", 0.0) != weight_decay:
            raise ValueError("continuation changed weight_decay")
        for key in (
            "format",
            "input_dim",
            "hidden_size",
            "action_dim",
            "learning_rate",
            "batch_size",
            "batches",
            "data",
        ):
            if original["config"][key] != config[key]:
                raise ValueError(f"continuation changed {key}")
        origin_epoch = original["epoch"]
        if epochs <= origin_epoch:
            raise ValueError("continuation must extend epoch budget")
    else:
        origin_epoch = 0
    freeze_spec(folder / "config.json", config)
    train, validation = (
        BalancedData(records, "train"),
        BalancedData(records, "supervised-validation"),
    )
    if train.names != validation.names:
        raise ValueError("missing validation condition")
    full = train.full_data() if full_epoch else None
    steps_per_epoch = math.ceil(len(full[0]) / batch_size) if full_epoch else batches
    torch.manual_seed(0)
    actor = actor_from_config(config).to(device)
    optimizer = (
        torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if architecture is not None
        else torch.optim.Adam(actor.parameters(), lr=learning_rate)
    )
    generator = torch.Generator().manual_seed(0)
    start = 0
    last = folder / "last.pt"
    if last.exists() or initial:
        state = torch.load(
            last if last.exists() else initial, map_location=device, weights_only=True
        )
        actor.load_state_dict(state["actor"])
        optimizer.load_state_dict(state["optimizer"])
        if last.exists() or continue_initial:
            if last.exists() and state["config"] != config:
                raise ValueError("training resume contract changed")
            start = state["epoch"]
            generator.set_state(state["sample_rng"].cpu())
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.startswith("cuda") and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    metrics_path = folder / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    metrics = [r for r in metrics if r["epoch"] <= start]
    previous_wall = metrics[-1]["wall_s"] if metrics else 0.0
    # Complete an interrupted scheduled evaluation before performing more updates.
    for epoch in range(interval, start + 1, interval):
        if epoch > origin_epoch:
            evaluate_checkpoint(folder / f"epoch-{epoch}.pt", f"{folder.name}-epoch-{epoch}")
    if start == epochs:
        return
    run = wandb.init(
        project="ecsai-deppo",
        name=os.environ.get("WANDB_NAME", f"multiscale-bc-{folder.name}"),
        mode=os.environ.get("WANDB_MODE", "offline"),
        dir=str(folder),
        config=config,
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    started = perf_counter()

    def checkpoint(epoch):
        return dict(
            config=config,
            epoch=epoch,
            actor=actor.state_dict(),
            optimizer=optimizer.state_dict(),
            sample_rng=generator.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
        )

    if not last.exists():
        atomic_save(checkpoint(start), last)
    try:
        for epoch in range(start + 1, epochs + 1):
            actor.train()
            losses, label_errors, norms = [], [], []
            order = torch.randperm(len(full[0]), generator=generator) if full_epoch else None
            sizes = []
            for batch in range(steps_per_epoch):
                if full_epoch:
                    indices = order[batch * batch_size : (batch + 1) * batch_size]
                    obs, labels, weights = [v[indices].to(device) for v in full]
                else:
                    obs, labels = [v.to(device) for v in train.sample(batch_size, generator)]
                    weights = 1.0
                target = safe_actions(labels)
                label_errors.append((labels - target).abs().max().item())
                dist, loc, _ = distribution(actor, obs)
                if loss_kind == "nll":
                    per_sample = -dist.log_prob(target)
                else:
                    center = target.new_tensor([0, 0, 0, 0, 0.5])
                    radius = target.new_tensor([5, 5, 5, 5, 0.45])
                    per_sample = (torch.tanh(loc) - (target - center) / radius).square().mean(-1)
                loss = (per_sample * weights).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite NLL; last.pt retains previous full epoch")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in actor.parameters()):
                    raise FloatingPointError("non-finite actor parameter")
                losses.append(loss.item())
                sizes.append(len(obs))
                norms.append(norm.item())
            actor.eval()
            row = dict(
                epoch=epoch,
                updates=epoch * steps_per_epoch,
                train_nll=float(np.average(losses, weights=sizes)),
                validation_nll=validation.validation_nll(actor, device),
                label_max_abs_error=max(label_errors),
                gradient_norm=float(np.mean(norms)),
                wall_s=previous_wall + perf_counter() - started,
            )
            if full_epoch:
                row["samples_this_epoch"] = sum(sizes)
                row["samples_seen"] = epoch * sum(sizes)
                row["updates_per_epoch"] = steps_per_epoch
            if loss_kind != "nll":
                row["train_mean_mse"] = row.pop("train_nll")
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError("non-finite training diagnostics")
            if epoch % interval == 0:
                atomic_save(checkpoint(epoch), folder / f"epoch-{epoch}.pt")
            metrics.append(row)
            write_json(metrics_path, metrics)
            atomic_save(checkpoint(epoch), last)
            with (folder / "metrics.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(metrics)
            run.log(row)
            print(f"FIT {folder.name} {epoch}/{epochs} nll={row['validation_nll']:.5f}", flush=True)
            if epoch % interval == 0:
                report = evaluate_checkpoint(
                    folder / f"epoch-{epoch}.pt", f"{folder.name}-epoch-{epoch}"
                )
                curves = {"epoch": epoch}
                for item in report["rows"]:
                    for key in (
                        "logical_success_rate",
                        "mean_success_e2e_s",
                        "mean_failed_elapsed_s",
                        "mean_resolution_time_s",
                        "deficit",
                    ):
                        curves[f"validation/{item['condition']}/{key}"] = item[key]
                run.log(curves)
    except Exception as exc:
        write_json(
            folder / "error.json",
            {
                "error": repr(exc),
                "last_complete_epoch": torch.load(last, map_location="cpu", weights_only=True)[
                    "epoch"
                ],
            },
        )
        raise
    finally:
        run.finish()


def make_conditions(source_root, smoke=False):
    conditions = {}
    if smoke:
        sources = [
            ("smoke2", ScenarioConfig.profile("smoke")),
            (
                "smoke3",
                ScenarioConfig.profile("smoke").model_copy(update={"clusters": 3, "caches": 3}),
            ),
        ]
        loads = [0.5]
    else:
        sources = [
            (
                size,
                ScenarioConfig.model_validate_json(
                    (source_root / f"scale-{size}/MAPPO/scenario.json").read_text()
                ),
            )
            for size in ("small", "medium", "large")
        ]
        loads = [0.25, 0.5, 0.75, 1.0, 1.25]
    for name, base in sources:
        for load in loads:
            cfg = ScenarioConfig.model_validate(
                base.model_dump()
                | {
                    "request_rate": base.request_rate * load / base.delivery_load,
                    "delivery_load": load,
                    "max_retries": 2,
                    "retry_delay_s": 0.1,
                    "scheduler_release": "window",
                }
                | ({"cycles": 4} if smoke else {})
            )
            a, _ = build_run(base, 42)
            b, _ = build_run(cfg, 42)
            if not np.allclose(
                [c.total_bandwidth_bytes_s for c in a.content.caches],
                [c.total_bandwidth_bytes_s for c in b.content.caches],
                rtol=1e-12,
            ):
                raise ValueError("load change altered physical capacity")
            conditions[f"{name}-rho{load:g}"] = cfg
    return conditions


def compare_conditions(output, conditions, checkpoint, baseline_reports, seed_split, count, tag):
    rows, reports = [], {}
    results = evaluation_batch(
        [
            (
                output / "evaluations" / tag / name,
                cfg.model_dump(),
                seeds(seed_split, i, count),
                "MAPPO-no-context",
                None,
                checkpoint,
            )
            for i, (name, cfg) in enumerate(conditions.items())
        ]
    )
    for name, result in zip(conditions, results, strict=True):
        ref_name, ref = reference_report(baseline_reports[name])
        rows.append(
            dict(
                condition=name,
                reference=ref_name,
                reference_success_rate=ref["mean"]["logical_success_rate"],
                reference_success_e2e_s=ref["mean"]["mean_success_e2e_s"],
                **result["mean"],
                **gate(result, ref),
                **paired_ci(result, ref),
            )
        )
        reports[name] = result
    rank = (
        max(r["deficit"] for r in rows),
        -float(np.mean([r["logical_success_rate"] for r in rows])),
    )
    report = dict(rows=rows, rank=list(rank), passed=all(r["passed"] for r in rows))
    write_json(output / "evaluations" / tag / "comparison.json", report)
    return report


def export_status(output, status, report, reason):
    payload = dict(
        status=status,
        reason=reason,
        failed_conditions=[r["condition"] for r in report["rows"] if not r["passed"]],
        **report,
    )
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = [
        f"# {status}",
        "",
        reason,
        "",
        "主评估为随机执行；每个条件单独验收。",
        "置信区间是配对评估 episode 的 bootstrap 区间，不代表跨训练种子稳定性。",
        "",
        "| 条件 | 参照基线 | 成功率 | 成功率差/pp | 成功总延迟/s | 延迟比 | 达标 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['condition']} | {r['reference']} | "
            f"{r['logical_success_rate'] * 100:.2f}% | {r['success_difference_pp']:+.2f} | "
            f"{r['mean_success_e2e_s']:.3f} | {r['latency_ratio']:.3f} | {r['passed']} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    render_report(output, report)
    import wandb

    with wandb.init(
        project="ecsai-deppo",
        name="multiscale-bc-verdict",
        mode=os.environ.get("WANDB_MODE", "offline"),
        dir=str(output),
        config={"status": status, "reason": reason},
    ) as run:
        columns = [
            "condition",
            "reference",
            "logical_success_rate",
            "mean_success_e2e_s",
            "mean_failed_elapsed_s",
            "mean_resolution_time_s",
            "mean_attempts",
            "success_difference_pp",
            "latency_ratio",
            "passed",
        ]
        run.log(
            {
                "comparison": wandb.Table(
                    columns=columns, data=[[r[c] for c in columns] for r in report["rows"]]
                ),
                "load_curves": wandb.Image(str(output / "load-curves.png")),
            }
        )
    write_json(output / "status.json", payload)
    print(f"STATUS {status}: {reason}", flush=True)


def render_report(output, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({r["condition"].split("-rho")[0] for r in report["rows"]})
    fig, axes = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 7), squeeze=False)
    for column, size in enumerate(sizes):
        rows = sorted(
            (r for r in report["rows"] if r["condition"].startswith(size + "-rho")),
            key=lambda r: float(r["condition"].split("-rho")[1]),
        )
        x = [float(r["condition"].split("-rho")[1]) for r in rows]
        for row_index, (student, baseline, scale, ylabel) in enumerate(
            [
                ("logical_success_rate", "reference_success_rate", 100, "Final success (%)"),
                ("mean_success_e2e_s", "reference_success_e2e_s", 1, "Successful E2E delay (s)"),
            ]
        ):
            ax = axes[row_index, column]
            ax.plot(x, [r[student] * scale for r in rows], "o-", label="BC base")
            ax.plot(x, [r[baseline] * scale for r in rows], "s--", label="Strongest baseline")
            ax.set(xlabel="Original delivery load ratio", ylabel=ylabel, title=size)
            ax.grid(alpha=0.2)
            ax.legend()
            if row_index == 0:
                ax.set_ylim(0, 105)
    fig.suptitle("Stochastic closed-loop evaluation with actual retries")
    fig.tight_layout()
    fig.savefig(output / "load-curves.png", dpi=160)
    fig.savefig(output / "load-curves.pdf")
    plt.close(fig)


def run_suite(source_root, output, workers=4, device="cpu", smoke=False):
    global EVAL_WORKERS
    EVAL_WORKERS = workers
    torch.set_num_threads(1)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions = make_conditions(source_root, smoke)
    spec = dict(
        format=FORMAT,
        conditions={n: c.model_dump() for n, c in conditions.items()},
        smoke=smoke,
        seed_bases=SEED_BASES,
        thresholds={"success_pp": 1, "latency_ratio": 1.05},
        teachers=TEACHERS,
        baselines=BASELINES,
    )
    freeze_spec(output / "spec.json", spec)
    if (output / "status.json").exists():
        print((output / "status.json").read_text(), flush=True)
        return
    write_json(output / "metadata.json", metadata(next(iter(conditions.values())), 0, "BC-v2"))
    selection_count, val_count = (1, 1) if smoke else (4, 4)
    baseline_reports, chosen, teacher_checks = {}, {}, {}
    # Independent conditions share a bounded pool of simulation workers.
    teacher_jobs = [
        (
            output / "teacher-selection" / name / teacher,
            cfg.model_dump(),
            seeds("selection", i, selection_count),
            *teacher_method(teacher),
            None,
        )
        for i, (name, cfg) in enumerate(conditions.items())
        for teacher in TEACHERS
    ]
    teacher_results = iter(evaluation_batch(teacher_jobs))
    for name in conditions:
        candidates = {teacher: next(teacher_results) for teacher in TEACHERS}
        chosen[name], _ = reference_report(candidates)
        freeze_spec(
            output / "teacher-selection" / name / "selected.json", {"teacher": chosen[name]}
        )
    baseline_jobs = [
        (
            output / "baselines-validation" / name / label,
            cfg.model_dump(),
            seeds("validation", i, val_count),
            method,
            ratio,
            None,
        )
        for i, (name, cfg) in enumerate(conditions.items())
        for label, (method, ratio) in BASELINES.items()
    ]
    baseline_results = iter(evaluation_batch(baseline_jobs))
    for name in conditions:
        baseline_reports[name] = {label: next(baseline_results) for label in BASELINES}
    chosen_jobs = [
        (
            output / "teacher-validation" / name,
            cfg.model_dump(),
            seeds("validation", i, val_count),
            *teacher_method(chosen[name]),
            None,
        )
        for i, (name, cfg) in enumerate(conditions.items())
    ]
    for name, selected_report in zip(conditions, evaluation_batch(chosen_jobs), strict=True):
        _, reference = reference_report(baseline_reports[name])
        teacher_checks[name] = gate(selected_report, reference)
    write_json(output / "teacher-checks.json", teacher_checks)
    records = []
    best_path = output / "best.pt"
    selection_path = output / "selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text())
        if not best_path.exists() or sha256(best_path) != saved["checkpoint_sha256"]:
            shutil.copyfile(saved["checkpoint"], best_path)

    def assess(checkpoint, tag):
        report = compare_conditions(
            output, conditions, checkpoint, baseline_reports, "validation", val_count, tag
        )
        old = json.loads(selection_path.read_text()) if selection_path.exists() else None
        if old is None or tuple(report["rank"]) < tuple(old["rank"]):
            temporary = best_path.with_suffix(".tmp")
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(best_path)
            write_json(
                selection_path,
                dict(report, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint)),
            )
        print(
            f"VALIDATE {tag} passed={sum(r['passed'] for r in report['rows'])}"
            f"/{len(conditions)} worst={report['rank'][0]:.3f}",
            flush=True,
        )
        return report

    for round_index in range(3):
        phase = output / f"round-{round_index}"
        phase.mkdir(parents=True, exist_ok=True)
        round_spec = phase / "round-spec.json"
        if round_spec.exists():
            failed = set(json.loads(round_spec.read_text())["conditions"])
        elif round_index:
            best = json.loads(selection_path.read_text())
            if best["passed"]:
                break
            failed = {r["condition"] for r in best["rows"] if not r["passed"]}
            teacher_failures = sorted(n for n in failed if not teacher_checks[n]["passed"])
            if teacher_failures:
                export_status(
                    output,
                    "base-not-ready",
                    best,
                    "停止补充：失败场景的冻结教师本身不达标：" + ", ".join(teacher_failures),
                )
                return
        else:
            failed = set(conditions)
        freeze_spec(round_spec, {"conditions": sorted(failed)})
        initial = phase / "initial.pt" if round_index else None
        if initial and not initial.exists():
            shutil.copyfile(best_path, initial)
        data_dir = phase / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for i, (name, cfg) in enumerate(conditions.items()):
            if name not in failed:
                continue
            splits = [("train", (2 if smoke else 16) if round_index == 0 else (2 if smoke else 32))]
            if round_index == 0:
                splits.append(("supervised-validation", 1 if smoke else 4))
            else:
                splits.append(("dagger", 1 if smoke else 16))
            for split, count in splits:
                for seed in seeds(split, i, count, round_index):
                    path = data_dir / f"{name}-{split}-{seed}.pt"
                    jobs.append(
                        (
                            cfg.model_dump(),
                            chosen[name],
                            str(path),
                            seed,
                            "train" if split == "dagger" else split,
                            name,
                            str(initial) if split == "dagger" else None,
                        )
                    )
        records.extend(collect_jobs(data_dir, jobs, workers))
        # Round input is frozen, including the selected initial model and dataset hashes.
        fit_phase(
            phase,
            records,
            initial,
            device,
            32 if smoke else 256,
            2 if smoke else (30 if round_index == 0 else 20),
            4 if smoke else 64,
            32 if smoke else 512,
            1 if smoke else 10,
            assess,
        )
    best = json.loads(selection_path.read_text())
    if not best["passed"]:
        export_status(output, "base-not-ready", best, "两轮补充后仍有场景未达到双门槛；未启动 RL。")
        return
    # Only now open the held-out test split; selected weights cannot change on resume.
    freeze_spec(output / "test-selection.json", {"checkpoint_sha256": sha256(best_path)})
    test_count = 2 if smoke else 30
    test_jobs = [
        (
            output / "baselines-test" / name / label,
            cfg.model_dump(),
            seeds("test", i, test_count),
            method,
            ratio,
            None,
        )
        for i, (name, cfg) in enumerate(conditions.items())
        for label, (method, ratio) in BASELINES.items()
    ]
    test_results = iter(evaluation_batch(test_jobs))
    test_baselines = {
        name: {label: next(test_results) for label in BASELINES} for name in conditions
    }
    final = compare_conditions(
        output, conditions, best_path, test_baselines, "test", test_count, "final-test"
    )
    export_status(
        output,
        "base-ready" if final["passed"] else "base-not-ready",
        final,
        "独立测试结束；测试集不再用于选择或修改模型。未启动 RL。",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (not args.smoke and args.source_root is None):
        parser.error("positive workers and source-root (unless smoke) are required")
    run_suite(args.source_root, args.output, args.workers, args.device, args.smoke)


if __name__ == "__main__":
    main()
