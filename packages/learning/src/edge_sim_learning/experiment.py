"""BenchMARL MAPPO orchestration, drained evaluation and auditable local logs."""

import copy
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
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

from .env import ACTION, OBS, SchedulingEnv
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


@torch.random.fork_rng(devices=[])
def evaluate(
    config,
    method="local",
    policy=None,
    seeds=None,
    fixed_ratio=None,
    exploration="deterministic",
    workers=1,
):
    """Run independent episodes in seed order, isolating each policy RNG in a process."""
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    seeds = list(seeds if seeds is not None else range(1_000_000_000, 1_000_000_010))
    if not seeds:
        raise ValueError("at least one evaluation episode is required")
    started = perf_counter()
    worker_count = min(workers, len(seeds))
    if worker_count == 1:
        result = _evaluate_serial(config, method, policy, seeds, fixed_ratio, exploration)
    else:
        # Never fork a CUDA training process. Only a detached CPU policy enters workers.
        cpu_policy = copy.deepcopy(policy).cpu().eval() if policy is not None else None
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_evaluation_worker,
            initargs=(config, method, cpu_policy, fixed_ratio, exploration),
        ) as pool:
            episodes = list(pool.map(_evaluate_seed, seeds))
        result = _evaluation_report(method, exploration, fixed_ratio, seeds, episodes)
    return result | {"workers": worker_count, "evaluation_wall_s": perf_counter() - started}


def _init_evaluation_worker(config, method, policy, fixed_ratio, exploration):
    global _evaluation_inputs
    torch.set_num_threads(1)
    _evaluation_inputs = (config, method, policy, fixed_ratio, exploration)


def _evaluate_seed(seed):
    config, method, policy, fixed_ratio, exploration = _evaluation_inputs
    return _evaluate_serial(config, method, policy, [seed], fixed_ratio, exploration)["episodes"][0]


@torch.random.fork_rng(devices=[])
def _evaluate_serial(
    config, method="local", policy=None, seeds=None, fixed_ratio=None, exploration="deterministic"
):
    if exploration not in {"deterministic", "stochastic"}:
        raise ValueError("unknown exploration mode")
    if fixed_ratio is not None and not 0.05 <= fixed_ratio <= 0.95:
        raise ValueError("fixed ratio must be in [0.05, 0.95]")
    torch.set_num_threads(1)
    seeds = list(seeds if seeds is not None else range(1_000_000_000, 1_000_000_010))
    if not seeds:
        raise ValueError("at least one evaluation episode is required")
    if policy is None and method not in {
        "random",
        "local",
        "forward",
        "queue-adaptive",
        "local-adaptive",
        "forward-adaptive",
    }:
        raise ValueError("learned evaluation requires a policy")
    episodes = []
    if policy is not None:
        policy = copy.deepcopy(policy).cpu().eval()
    for seed in seeds:
        if exploration == "stochastic":
            torch.manual_seed(seed)
        env_type = SchedulingEnv
        if getattr(policy, "observation_profile", None) == "local-context-v2":
            from .local_context import LocalContextEnv

            env_type = LocalContextEnv
        env = env_type(config, evaluation=True, method=method)
        started = perf_counter()
        inference_s = 0
        inference_calls = 0
        try:
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 777]))

            def infer(observations, env=env):
                nonlocal inference_s, inference_calls
                td = TensorDict(
                    {
                        "agents": TensorDict(
                            {
                                "observation": torch.tensor(
                                    np.stack([observations[a] for a in env.possible_agents])
                                )
                            },
                            [config.clusters],
                        ),
                        "state": torch.tensor(env.state()),
                    },
                    [],
                )
                inference_start = perf_counter()
                with (
                    torch.no_grad(),
                    set_exploration_type(
                        ExplorationType.DETERMINISTIC
                        if exploration == "deterministic"
                        else ExplorationType.RANDOM
                    ),
                ):
                    action = policy(td)["agents", "action"].numpy()
                inference_s += perf_counter() - inference_start
                inference_calls += 1
                return dict(zip(env.possible_agents, action, strict=True))

            def rule_actions(_, env=env, rng=rng):
                action = np.tile(
                    [0, 0, 0, -5 if method in {"local", "local-adaptive"} else 5, 0.5],
                    (config.clusters, 1),
                ).astype(np.float32)
                if method == "random" and fixed_ratio is None:
                    action[:, 4] = rng.uniform(0.4, 0.6, config.clusters)
                if method in {"queue-adaptive", "local-adaptive", "forward-adaptive"}:
                    # Allocation is independent of the selected forwarding rule.
                    for i, agent in enumerate(env.possible_agents):
                        pools = [
                            p for p in env.last_view.pools if env.cache_clusters[p.node_id] == agent
                        ]
                        back = sum(p.remaining_bytes for p in pools if p.kind == "backhaul")
                        total = sum(p.remaining_bytes for p in pools)
                        action[i, 4] = np.clip(back / total, 0.05, 0.95) if total else 0.5
                if fixed_ratio is not None:
                    action[:, 4] = fixed_ratio
                return dict(zip(env.possible_agents, action, strict=True))

            while env.agents:
                actions = infer(obs) if policy is not None else rule_actions(obs)
                if method in {"random", "queue-adaptive"} and policy is None:
                    env.begin(actions, policy="random")
                obs, _, _, _, _ = env.step(actions)
            truncated_metrics = env.metrics()
            metrics = env.drain(
                infer if policy is not None else rule_actions if config.max_retries else None,
                policy="random"
                if policy is None and method in {"random", "queue-adaptive"}
                else "threshold",
            )
            if config.episode_loads and config.max_retries:
                outcomes = {r["request_id"]: r for r in env.retry_records()}
                for index, (start, end, load, _) in enumerate(config.load_phases()):
                    rows = [
                        outcomes[r.id]
                        for r in env.run.content.requests
                        if start <= r.arrival_s < end
                    ]
                    succeeded = [r for r in rows if r["status"] == "SUCCEEDED"]
                    prefix = f"phase_{index}_rho{load:g}/"
                    metrics[prefix + "requests"] = len(rows)
                    metrics[prefix + "logical_success_rate"] = len(succeeded) / max(1, len(rows))
                    metrics[prefix + "mean_success_e2e_s"] = sum(
                        r["total_elapsed_s"] for r in succeeded
                    ) / max(1, len(succeeded))
                    metrics[prefix + "mean_resolution_time_s"] = sum(
                        r["total_elapsed_s"] for r in rows
                    ) / max(1, len(rows))
            episodes.append(
                metrics
                | ({"retry_outcomes": env.retry_records()} if config.max_retries else {})
                | {
                    "seed": seed,
                    "workload": env.workload,
                    "episode_return": env.episode_return,
                    "unscaled_episode_return": env.episode_return / config.reward_scale,
                    "unfinished_at_truncation": truncated_metrics["unfinished"],
                    "wall_s": perf_counter() - started,
                    "inference_wall_s": inference_s,
                    "policy_joint_calls": inference_calls,
                    "policy_agent_calls": inference_calls * config.clusters,
                }
            )
        finally:
            env.close()
    return _evaluation_report(method, exploration, fixed_ratio, seeds, episodes)


def _evaluation_report(method, exploration, fixed_ratio, seeds, episodes):
    means = {
        key: float(np.mean([row[key] for row in episodes]))
        for key in episodes[0]
        if key not in {"seed", "workload", "retry_outcomes"}
    }
    return {
        "method": method,
        "exploration": exploration,
        "fixed_ratio": fixed_ratio,
        "seeds": seeds,
        "mean": means,
        "episodes": episodes,
    }


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
        "adaptation": (
            "DD-adapted: fixed FIFO request slots, padded local request features, "
            "bounded per-request scores thresholded at zero plus bandwidth ratio; "
            "window release, MAPPO training, no GRU; not exact original DD reproduction."
            if method == "DD-adapted"
            else ADAPTATION
        ),
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
    def __init__(
        self,
        output,
        scenario,
        method,
        seed,
        mode,
        eval_interval,
        eval_episodes,
        eval_stochastic=False,
        eval_workers=1,
    ):
        super().__init__()
        self.output = str(output)
        self.scenario, self.method, self.seed = scenario, method, seed
        self.mode, self.eval_interval, self.eval_episodes = mode, eval_interval, eval_episodes
        self.eval_stochastic = eval_stochastic
        self.eval_workers = eval_workers
        self.best = (-1.0, float("-inf"))
        self.best_stochastic = (-1.0, float("-inf"))

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
            name=os.environ.get("ECSAI_RUN_NAME", f"{self.method}-seed{self.seed}"),
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
        if getattr(self, "local_context", False):
            loads = batch["next", "metrics", "delivery_load"]
            for load in self.load_mix:
                metrics[f"load/rho{load:g}/frames"] = (
                    torch.isclose(loads, loads.new_tensor(load)).sum().item()
                )
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
            for dim in range(actions.shape[-1]):
                x = actions[..., agent, dim]
                metrics[f"actor_{agent}/action_{dim}_mean"] = x.mean().item()
                metrics[f"actor_{agent}/action_{dim}_std"] = x.std(unbiased=False).item()
                low, high = (
                    (0.05, 0.95)
                    if dim == actions.shape[-1] - 1
                    else ((-1, 1) if self.method == "DD-adapted" else (-5, 5))
                )
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
            "training_options": self.training_options,
            "action_dim": self.scenario.scheduler_capacity + 2
            if self.method == "DD-adapted"
            else ACTION,
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
            "best_stochastic": self.best_stochastic,
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
            if getattr(self, "local_context", False):
                self.evaluate_contextual()
                self.checkpoint("last.pt")
                return
            result = evaluate(
                self.scenario,
                self.method,
                exp.policy,
                range(1_000_000_000, 1_000_000_000 + self.eval_episodes),
                workers=self.eval_workers,
            )
            self.emit(
                {"eval/" + k: v for k, v in result["mean"].items()}
                | {
                    "eval/evaluation_wall_s": result["evaluation_wall_s"],
                }
            )
            (Path(self.output) / f"evaluation-{exp.total_frames}.json").write_text(
                json.dumps(result, indent=2)
            )
            score = (result["mean"]["success_rate"], -result["mean"]["mean_latency_s"])
            if score > self.best:
                self.best = score
                self.checkpoint("best.pt")
            if self.eval_stochastic:
                sampled = evaluate(
                    self.scenario,
                    self.method,
                    exp.policy,
                    result["seeds"],
                    exploration="stochastic",
                    workers=self.eval_workers,
                )
                self.emit(
                    {"eval_stochastic/" + k: v for k, v in sampled["mean"].items()}
                    | {
                        "eval_stochastic/evaluation_wall_s": sampled["evaluation_wall_s"],
                    }
                )
                (Path(self.output) / f"evaluation-stochastic-{exp.total_frames}.json").write_text(
                    json.dumps(sampled, indent=2)
                )
                sampled_score = (
                    sampled["mean"]["success_rate"],
                    -sampled["mean"]["mean_latency_s"],
                )
                if sampled_score > self.best_stochastic:
                    self.best_stochastic = sampled_score
                    self.checkpoint("best-stochastic.pt")
        self.checkpoint("last.pt")

    def evaluate_contextual(self, advance_iteration=True):
        from .multiscale_bc import evaluation_slot

        self.experiment.policy.observation_profile = "local-context-v2"
        reports = {}
        with evaluation_slot():
            for index, _load in enumerate((self.scenario.delivery_load,)):
                cfg = self.scenario.model_copy(
                    update={
                        "episode_loads": self.load_mix,
                    }
                )
                result = evaluate(
                    cfg,
                    self.method,
                    self.experiment.policy,
                    range(
                        1_100_000_000 + index * 10000,
                        1_100_000_000 + index * 10000 + self.eval_episodes,
                    ),
                    exploration="stochastic",
                    workers=self.eval_workers,
                )
                reports["within-episode"] = result
                self.emit({f"eval/within-episode/{k}": v for k, v in result["mean"].items()})
        means = {
            k: float(np.mean([r["mean"][k] for r in reports.values()]))
            for k in (
                "logical_success_rate",
                "mean_success_e2e_s",
                "mean_failed_elapsed_s",
                "mean_resolution_time_s",
                "mean_attempts",
                "episode_return",
            )
        }
        self.emit({"eval/" + k: v for k, v in means.items()})
        path = Path(self.output) / f"evaluation-context-{self.experiment.total_frames}.json"
        path.write_text(json.dumps(dict(mean=means, conditions=reports), indent=2))
        score = (means["logical_success_rate"], -means["mean_success_e2e_s"])
        if score > self.best:
            self.best = self.best_stochastic = score
            self.checkpoint("best.pt", advance_iteration=advance_iteration)
            self.checkpoint("best-stochastic.pt", advance_iteration=advance_iteration)
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
    normalize_advantage=False,
    hidden_size=128,
    context_size=64,
    initial_std=None,
    eval_stochastic=False,
    eval_workers=1,
    actor_init=None,
    head_only=False,
    local_context=False,
    load_mix=(),
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (
        min(
            episodes,
            workers,
            frames_per_batch,
            epochs,
            minibatch,
            eval_interval,
            eval_episodes,
            eval_workers,
        )
        < 1
    ):
        raise ValueError("training counts must be positive")
    if method not in {"DEPPO-adapted", "MAPPO-no-context", "DD-adapted"}:
        raise ValueError("unknown learning method")
    if actor_init is not None and resume is not None:
        raise ValueError("choose base initialization or checkpoint resume, not both")
    if (actor_init is not None or head_only) and method != "MAPPO-no-context":
        raise ValueError("base adaptation currently supports MAPPO-no-context")
    if head_only and actor_init is None and resume is None:
        raise ValueError("head-only training requires a base actor or resumed checkpoint")
    if local_context and (method != "MAPPO-no-context" or hidden_size != 256):
        raise ValueError("local-context RL requires MAPPO-no-context and hidden_size=256")
    if load_mix and (not local_context or any(x <= 0 for x in load_mix)):
        raise ValueError("load mixture requires local-context RL and positive loads")
    if method == "DD-adapted":
        scenario = scenario.model_copy(update={"scheduler_release": "window"})
    action_dim = scenario.scheduler_capacity + 2 if method == "DD-adapted" else ACTION
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
        num_cells=[hidden_size, hidden_size],
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.ReLU,
    )
    callback = RunLog(
        output,
        scenario,
        method,
        seed,
        mode,
        eval_interval,
        eval_episodes,
        eval_stochastic,
        eval_workers,
    )
    callback.local_context = local_context
    callback.load_mix = tuple(load_mix)
    training_options = dict(
        normalize_advantage=normalize_advantage,
        hidden_size=hidden_size,
        context_size=context_size,
        initial_std=initial_std,
    )
    if local_context:
        training_options.update(
            local_context=True,
            load_mix=list(load_mix),
            fixed_scale=0.1,
            rl_dropout=False,
            load_schedule="within-episode",
        )
    payload = torch.load(resume, map_location=device, weights_only=False) if resume else None
    if head_only and payload and not payload.get("training_options", {}).get("head_only", False):
        raise ValueError("resume cannot change full training into head-only adaptation")
    if actor_init is not None:
        training_options.update(
            actor_init_sha256=hashlib.sha256(Path(actor_init).read_bytes()).hexdigest(),
            head_only=head_only,
        )
    elif payload and "actor_init_sha256" in payload.get("training_options", {}):
        training_options.update(
            actor_init_sha256=payload["training_options"]["actor_init_sha256"],
            head_only=payload["training_options"]["head_only"],
        )
        head_only = training_options["head_only"]
    callback.training_options = training_options
    config = metadata(scenario, seed, method) | {
        "training_options": training_options,
        "experiment": cfg.__dict__,
        "algorithm": algorithm.__dict__,
        "episodes": episodes,
        "workers": workers,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "eval_stochastic": eval_stochastic,
        "eval_workers": eval_workers,
        "wandb_mode": mode,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    if payload:
        legacy = dict(normalize_advantage=False, hidden_size=128, context_size=64, initial_std=None)
        if payload.get("training_options", legacy) != training_options:
            raise ValueError("checkpoint training options mismatch")
        if payload.get("action_dim") != action_dim:
            raise ValueError("checkpoint action dimension mismatch")
        if (
            type(scenario).model_validate(payload["scenario"]) != scenario
            or payload["method"] != method
        ):
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
        task=ContentTask(scenario, episode_start, method, local_context, tuple(load_mix)),
        algorithm_config=algorithm,
        model_config=ContextConfig(
            use_context=method == "DEPPO-adapted",
            input_dim=OBS + 4 * (scenario.scheduler_capacity + 1)
            if method == "DD-adapted"
            else 21
            if local_context
            else OBS,
            hidden_size=hidden_size,
            context_size=context_size,
            initial_std=initial_std,
            actor_init=str(Path(actor_init).resolve()) if actor_init is not None else None,
            head_only=head_only,
            local_context=local_context,
        ),
        critic_model_config=critic,
        seed=seed,
        config=cfg,
        callbacks=[callback],
    )
    for loss in experiment.losses.values():
        loss.normalize_advantage = normalize_advantage
        loss.normalize_advantage_exclude_dims = (-2,)
    if local_context:
        experiment.policy.observation_profile = "local-context-v2"
    try:
        if resume:
            experiment.load_state_dict(payload["experiment"])
            for g, items in experiment.optimizers.items():
                for k, optimizer in items.items():
                    optimizer.load_state_dict(payload["optimizers"][g][k])
            for field, filename in (("best", "best.pt"), ("best_stochastic", "best-stochastic.pt")):
                previous_best = Path(resume).resolve().parent / filename
                if field in payload and previous_best.exists():
                    best_payload = torch.load(previous_best, map_location="cpu", weights_only=False)
                    if best_payload.get(field) == payload[field]:
                        setattr(callback, field, payload[field])
                        if previous_best != output / filename:
                            shutil.copy2(previous_best, output / filename)
            restore_rng_states(payload)
            experiment.collector.update_policy_weights_()
        callback.checkpoint("last.pt", advance_iteration=False)
        if local_context and not resume:
            callback.checkpoint("initial.pt", advance_iteration=False)
            callback.evaluate_contextual(advance_iteration=False)
            initial_payload = torch.load(
                output / "initial.pt", map_location="cpu", weights_only=False
            )
            restore_rng_states(initial_payload)
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
        name=os.environ.get("ECSAI_RUN_NAME", method),
        config=meta,
    )
    try:
        for row in rows:
            run.log(row)
    finally:
        run.finish()
