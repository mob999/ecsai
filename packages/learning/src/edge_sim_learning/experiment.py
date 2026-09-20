"""BenchMARL MAPPO orchestration, drained evaluation and auditable local logs."""

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import MlpConfig
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .env import ACTION, SchedulingEnv
from .model import ContextConfig
from .scenario import build_run
from .stability import StableExperiment
from .torch_env import ContentTask

ADAPTATION = (
    "DEPPO-adapted v2: five actions, synthetic whole-object workload, "
    "per-cache shared backhaul/delivery budget, FIFO pools, LRU, "
    "modified team reward; not paper table replication."
)


def restore_rng_states(payload):
    """Checkpoint map_location may move generator states; restore from CPU bytes."""
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])


def evaluate(config, method="local", policy=None, seeds=None, fixed_ratio=None):
    if fixed_ratio is not None and not 0.05 <= fixed_ratio <= 0.95:
        raise ValueError("fixed ratio must be in [0.05, 0.95]")
    torch.set_num_threads(1)
    seeds = list(seeds if seeds is not None else range(1_000_000_000, 1_000_000_010))
    if not seeds:
        raise ValueError("at least one evaluation episode is required")
    if policy is None and method not in {"random", "local", "forward", "queue-adaptive"}:
        raise ValueError("learned evaluation requires a policy")
    episodes = []
    if policy is not None:
        policy = copy.deepcopy(policy).cpu().eval()
    for seed in seeds:
        env = SchedulingEnv(config, evaluation=True)
        started = perf_counter()
        inference_s = 0
        try:
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 777]))
            while env.agents:
                if policy is not None:
                    td = TensorDict(
                        {
                            "agents": TensorDict(
                                {
                                    "observation": torch.tensor(
                                        np.stack([obs[a] for a in env.possible_agents])
                                    )
                                },
                                [config.clusters],
                            ),
                            "state": torch.tensor(env.state()),
                        },
                        [],
                    )
                    inference_start = perf_counter()
                    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                        action = policy(td)["agents", "action"].numpy()
                    inference_s += perf_counter() - inference_start
                else:
                    action = np.tile(
                        [0, 0, 0, -5 if method == "local" else 5, 0.5], (config.clusters, 1)
                    ).astype(np.float32)
                    if method == "random" and fixed_ratio is None:
                        action[:, 4] = rng.uniform(0.4, 0.6, config.clusters)
                    if method == "queue-adaptive":
                        # Same per-request Bernoulli forwarding as Random, only allocation differs.
                        for i, agent in enumerate(env.possible_agents):
                            pools = [
                                p
                                for p in env.last_view.pools
                                if env.cache_clusters[p.node_id] == agent
                            ]
                            back = sum(p.remaining_bytes for p in pools if p.kind == "backhaul")
                            total = sum(p.remaining_bytes for p in pools)
                            action[i, 4] = np.clip(back / total, 0.05, 0.95) if total else 0.5
                    if fixed_ratio is not None:
                        action[:, 4] = fixed_ratio
                actions = dict(zip(env.possible_agents, action, strict=True))
                if method in {"random", "queue-adaptive"} and policy is None:
                    env.begin(actions, policy="random")
                obs, _, _, _, _ = env.step(actions)
            truncated_metrics = env.metrics()
            metrics = env.drain()
            episodes.append(
                metrics
                | {
                    "seed": seed,
                    "workload": env.workload,
                    "episode_return": env.episode_return,
                    "unfinished_at_truncation": truncated_metrics["unfinished"],
                    "wall_s": perf_counter() - started,
                    "inference_wall_s": inference_s,
                }
            )
        finally:
            env.close()
    means = {
        key: float(np.mean([row[key] for row in episodes]))
        for key in episodes[0]
        if key not in {"seed", "workload"}
    }
    return {"method": method, "seeds": seeds, "mean": means, "episodes": episodes}


def metadata(config, seed, method):
    code = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    )
    _, workload = build_run(config, seed)
    source_root = Path(__file__).resolve().parents[4]
    source_hash = hashlib.sha256()
    for path in sorted((source_root / "packages").rglob("*.py")):
        source_hash.update(str(path.relative_to(source_root)).encode())
        source_hash.update(path.read_bytes())
    return {
        "scenario": config.model_dump(),
        "seed": seed,
        "method": method,
        "adaptation": ADAPTATION,
        "code_revision": code,
        "source_sha256": source_hash.hexdigest(),
        "working_tree_dirty": dirty,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in [
                "benchmarl",
                "torch",
                "torchrl",
                "tensordict",
                "pettingzoo",
                "wandb",
                "simgrid",
            ]
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "workload": workload,
    }


class InferenceTimer:
    def __init__(self):
        self.elapsed = 0.0

    def before(self, module, inputs):
        self.started = perf_counter()

    def after(self, module, inputs, output):
        self.elapsed += perf_counter() - self.started


class RunLog(Callback):
    def __init__(self, output, scenario, method, seed, mode, eval_interval, eval_episodes):
        super().__init__()
        self.output = str(output)
        self.scenario, self.method, self.seed = scenario, method, seed
        self.mode, self.eval_interval, self.eval_episodes = mode, eval_interval, eval_episodes
        self.best = (-1.0, float("-inf"))

    def on_setup(self):
        import wandb

        self.experiment.test_env.close()  # spec probe is not a fifth sampling worker
        self.started = self.collection_started = perf_counter()
        self.inference_timer = InferenceTimer()
        self.experiment.collector.policy.register_forward_pre_hook(self.inference_timer.before)
        self.experiment.collector.policy.register_forward_hook(self.inference_timer.after)
        self.run = wandb.init(
            project="ecsai-deppo",
            entity=os.environ.get("WANDB_ENTITY"),
            mode=self.mode,
            dir=self.output,
            name=f"{self.method}-seed{self.seed}",
            config=metadata(self.scenario, self.seed, self.method),
        )
        self.run.define_metric("env_steps")
        self.run.define_metric("wall_s")
        self.run.define_metric("*", step_metric="env_steps")
        path = Path(self.output) / "metrics.csv"
        if not path.exists():
            with path.open("w") as f:
                csv.writer(f).writerow(["env_steps", "agent_steps", "wall_s", "metric", "value"])

    def emit(self, metrics):
        frames = self.experiment.total_frames
        wall = perf_counter() - self.started
        values = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"non-finite metrics: {values}")
        with (Path(self.output) / "metrics.csv").open("a") as f:
            csv.writer(f).writerows(
                [frames, frames * self.scenario.clusters, wall, key, val]
                for key, val in values.items()
            )
        self.run.log(
            values
            | {"env_steps": frames, "agent_steps": frames * self.scenario.clusters, "wall_s": wall}
        )

    def on_batch_collected(self, batch):
        elapsed = perf_counter() - self.collection_started
        metrics = {
            "inference_wall_s": self.inference_timer.elapsed,
            "sampling_steps_s": batch.numel() / elapsed,
            "collection_wall_s": elapsed,
        }
        for key, value in batch["next", "metrics"].items():
            metrics[key] = value.float().mean().item()
        metrics["requests_processed_s"] = (
            batch["next", "metrics", "window_resolved"].sum().item() / elapsed
        )
        done = batch["next", "done"].squeeze(-1)
        if done.any():
            metrics["episode_return"] = (
                batch["next", "metrics", "episode_return"].squeeze(-1)[done].mean().item()
            )
        # Residual includes tensor packing and collector work; do not mislabel as pure inference.
        metrics["collector_overhead_wall_s"] = max(
            0,
            elapsed
            - batch["next", "metrics", "window_wall_s"].squeeze(-1).max(0).values.sum().item(),
        )
        actions = batch["agents", "action"].detach()
        for agent in range(self.scenario.clusters):
            for dim in range(ACTION):
                x = actions[..., agent, dim]
                metrics[f"actor_{agent}/action_{dim}_mean"] = x.mean().item()
                metrics[f"actor_{agent}/action_{dim}_std"] = x.std(unbiased=False).item()
                low, high = (-5, 5) if dim < 4 else (0.05, 0.95)
                metrics[f"actor_{agent}/action_{dim}_saturation"] = (
                    ((x - low < 0.05 * (high - low)) | (high - x < 0.05 * (high - low)))
                    .float()
                    .mean()
                    .item()
                )
                for param in ("loc", "scale"):
                    metrics[f"actor_{agent}/{param}_{dim}"] = (
                        batch["agents", param][..., agent, dim].mean().item()
                    )
        self.emit({"train/" + k: v for k, v in metrics.items()})
        self.inference_timer.elapsed = 0.0
        self.train_started = perf_counter()

    def checkpoint(self, name, advance_iteration=True):
        exp = self.experiment
        state = exp.state_dict()
        state["state"]["n_iters_performed"] = exp.n_iters_performed + int(advance_iteration)
        payload = {
            "action_dim": ACTION,
            "format_version": 2,
            "experiment": state,
            "optimizers": {
                g: {k: o.state_dict() for k, o in items.items()}
                for g, items in exp.optimizers.items()
            },
            "policy": copy.deepcopy(exp.policy).cpu(),
            "scenario": self.scenario.model_dump(),
            "method": self.method,
            "seed": self.seed,
            "best": self.best,
            "workers": exp.config.on_policy_n_envs_per_worker,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        path = Path(self.output) / name
        torch.save(payload, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def on_train_end(self, training_td, group):
        losses = {
            "train/" + str(k): v.float().mean().item()
            for k, v in training_td.items(True, True)
            if torch.is_tensor(v)
        }
        for key in ("actor_updates", "kl_early_stop"):
            if key in training_td.keys():
                losses["train/" + key] = training_td[key].float().max().item()
        losses["train/training_wall_s"] = perf_counter() - self.train_started
        self.emit(losses)
        exp = self.experiment
        self.checkpoint("last.pt")
        if (
            exp.total_frames % self.eval_interval == 0
            or exp.total_frames >= exp.config.max_n_frames
        ):
            result = evaluate(
                self.scenario,
                self.method,
                exp.policy,
                range(1_000_000_000, 1_000_000_000 + self.eval_episodes),
            )
            self.emit({"eval/" + k: v for k, v in result["mean"].items()})
            (Path(self.output) / f"evaluation-{exp.total_frames}.json").write_text(
                json.dumps(result, indent=2)
            )
            score = (result["mean"]["success_rate"], -result["mean"]["mean_latency_s"])
            if score > self.best:
                self.best = score
                self.checkpoint("best.pt")
        self.checkpoint("last.pt")
        self.collection_started = perf_counter()


def train(
    scenario,
    output,
    method="DEPPO-adapted",
    seed=0,
    episodes=2048,
    workers=4,
    frames_per_batch=512,
    epochs=5,
    minibatch=512,
    device="cpu",
    mode="offline",
    eval_interval=8192,
    eval_episodes=10,
    resume=None,
    learning_rate=1e-4,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (
        min(episodes, workers, frames_per_batch, epochs, minibatch, eval_interval, eval_episodes)
        < 1
    ):
        raise ValueError("training counts must be positive")
    if method not in {"DEPPO-adapted", "MAPPO-no-context"}:
        raise ValueError("unknown learning method")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requires an available NVIDIA GPU")
    if episodes * scenario.cycles % frames_per_batch or frames_per_batch % (
        workers * scenario.cycles
    ):
        raise ValueError(
            "episode budget must divide into whole collection batches; "
            "batch must contain whole episodes for every worker"
        )
    if eval_interval % frames_per_batch:
        raise ValueError("evaluation interval must be a multiple of batch size")
    torch.set_num_threads(1)
    cfg = ExperimentConfig.get_from_yaml()
    cfg.sampling_device, cfg.train_device, cfg.buffer_device = "cpu", device, device
    cfg.share_policy_params, cfg.parallel_collection = False, False
    cfg.lr, cfg.gamma = learning_rate, 0.99
    cfg.clip_grad_norm, cfg.clip_grad_val = True, 0.5
    cfg.max_n_frames, cfg.max_n_iters = episodes * scenario.cycles, None
    cfg.on_policy_collected_frames_per_batch = frames_per_batch
    cfg.on_policy_n_envs_per_worker = workers
    cfg.on_policy_n_minibatch_iters, cfg.on_policy_minibatch_size = epochs, minibatch
    cfg.evaluation, cfg.render, cfg.create_json = False, False, False
    # Custom drained evaluation runs sequentially; only one spec-probe environment is needed.
    cfg.evaluation_episodes = 1
    cfg.loggers, cfg.save_folder = ["csv"], str(output)
    algorithm = MappoConfig.get_from_yaml()
    algorithm.lmbda, algorithm.clip_epsilon, algorithm.share_param_critic = 0.95, 0.1, True
    algorithm.entropy_coef = 0.001
    critic = MlpConfig(
        num_cells=[128, 128], layer_class=torch.nn.Linear, activation_class=torch.nn.ReLU
    )
    callback = RunLog(output, scenario, method, seed, mode, eval_interval, eval_episodes)
    config = metadata(scenario, seed, method) | {
        "experiment": cfg.__dict__,
        "algorithm": algorithm.__dict__,
        "episodes": episodes,
        "workers": workers,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "wandb_mode": mode,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    payload = torch.load(resume, map_location=device, weights_only=False) if resume else None
    if payload:
        if payload.get("action_dim") != ACTION:
            raise ValueError("checkpoint action dimension mismatch: v2 requires five actions")
        if payload["scenario"] != scenario.model_dump() or payload["method"] != method:
            raise ValueError("checkpoint scenario/method mismatch")
        if payload.get("workers", workers) != workers or payload["seed"] != seed:
            raise ValueError("resume requires the original worker count and training seed")
        if payload["experiment"]["state"]["total_frames"] >= cfg.max_n_frames:
            raise ValueError("resume episode budget must exceed completed training")
    episode_start = (
        payload["experiment"]["state"]["total_frames"] // (workers * scenario.cycles)
        if payload
        else 0
    )
    experiment = StableExperiment(
        task=ContentTask(scenario, episode_start),
        algorithm_config=algorithm,
        model_config=ContextConfig(use_context=method == "DEPPO-adapted"),
        critic_model_config=critic,
        seed=seed,
        config=cfg,
        callbacks=[callback],
    )
    try:
        if resume:
            experiment.load_state_dict(payload["experiment"])
            for g, items in experiment.optimizers.items():
                for k, optimizer in items.items():
                    optimizer.load_state_dict(payload["optimizers"][g][k])
            previous_best = Path(resume).resolve().parent / "best.pt"
            if previous_best.exists():
                best_payload = torch.load(previous_best, map_location="cpu", weights_only=False)
                if best_payload["best"] == payload["best"]:
                    callback.best = payload["best"]
                    if previous_best != output / "best.pt":
                        shutil.copy2(previous_best, output / "best.pt")
            restore_rng_states(payload)
            experiment.collector.update_policy_weights_()
        callback.checkpoint("last.pt", advance_iteration=False)
        experiment.run()
    finally:
        experiment.close()
        if hasattr(callback, "run"):
            callback.run.finish()
    return output


def save_report(output, config, seed, method, mode, rows):
    """Use the same local/W&B telemetry contract for standalone evaluation and benchmarks."""
    import wandb

    output = Path(output).resolve()
    meta = metadata(config, seed, method)
    (output / "config.json").write_text(json.dumps(meta, indent=2))
    with (output / "metrics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "metric", "value"])
        for index, row in enumerate(rows):
            writer.writerows([index, key, value] for key, value in row.items())
    run = wandb.init(
        project="ecsai-deppo",
        entity=os.environ.get("WANDB_ENTITY"),
        mode=mode,
        dir=str(output),
        name=method,
        config=meta,
    )
    try:
        for row in rows:
            run.log(row)
    finally:
        run.finish()
