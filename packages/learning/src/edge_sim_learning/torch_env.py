"""Vector sampling over SDK workers; no second layer of process vectorization."""

from time import perf_counter

import numpy as np
import torch
from benchmarl.environments.common import TaskClass
from edge_sim import BatchRunner, SDKError
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, PettingZooWrapper

from .env import SchedulingEnv
from .scenario import ScenarioConfig


def contextual_env(config, load_mix=(), **kwargs):
    from .local_context import LocalContextEnv

    class MixedContextEnv(LocalContextEnv):
        def __init__(self, *args, **kwargs):
            from gymnasium.spaces import Box

            super().__init__(*args, **kwargs)
            self.state_space = Box(-np.inf, np.inf, (21 * self.config.clusters,), np.float32)

        def prepare_reset(self, seed=None):
            generation = 0 if seed is not None else self.generation
            if load_mix:
                load = load_mix[(generation + self.slot) % len(load_mix)]
                self.config = config.model_copy(
                    update={
                        "delivery_load": load,
                        "request_rate": config.request_rate * load / config.delivery_load,
                    }
                )
            super().prepare_reset(seed)

        def state(self):
            values = self._observations()
            return np.concatenate([values[a] for a in self.possible_agents]).astype(np.float32)

    return MixedContextEnv(config, **kwargs)


class ContentBatchEnv(EnvBase):
    def __init__(
        self,
        config,
        workers=4,
        seed=0,
        episode_start=0,
        method="DEPPO-adapted",
        local_context=False,
        load_mix=(),
    ):
        super().__init__(device="cpu", batch_size=[workers])
        self.runner = BatchRunner(workers=workers)
        self.envs = [
            (
                contextual_env(
                    config, load_mix, seed=seed, runner=self.runner, slot=i, method=method
                )
                if local_context
                else SchedulingEnv(config, seed=seed, runner=self.runner, slot=i, method=method)
            )
            for i in range(workers)
        ]
        for env in self.envs:
            env.generation = episode_start
        self._prepare_resets(self.envs)
        self.group_map = {"agents": self.envs[0].possible_agents}
        self.wrappers = [
            PettingZooWrapper(e, group_map=self.group_map, return_state=True, use_mask=False)
            for e in self.envs
        ]
        for env in self.envs:
            env.generation = episode_start
        prototype = self.wrappers[0]
        self.metric_keys = sorted(
            self.envs[0].last_metrics.keys()
            | {
                "reward",
                "paper_reward",
                "business_reward",
                "episode_return",
                "simulation_wall_s",
                "ipc_wall_s",
                "rpc_overhead_wall_s",
                "encoding_wall_s",
                "window_wall_s",
                "window_completed",
                "window_resolved",
            }
        )
        if local_context:
            self.metric_keys.append("delivery_load")
        observation = prototype.observation_spec.clone()
        observation["metrics"] = Composite({k: Unbounded(shape=(1,)) for k in self.metric_keys})
        self.observation_spec = observation.expand(workers)
        self.action_spec = prototype.full_action_spec.expand(workers)
        self.reward_spec = prototype.full_reward_spec.expand(workers)
        self.done_spec = prototype.full_done_spec.expand(workers)
        self.previous = [None] * workers
        self.is_closed = False

    def _prepare_resets(self, envs):
        for env in envs:
            env.prepare_reset()
        pending = [env for env in envs if env.session is None]
        self.runner.submit_many(env.run for env in pending)
        for env in pending:
            env.session = self.runner.session(env.run_id)

    def _pack_step(self, env, observations, rewards, terminated, truncated):
        # The fixed-agent contract has no dynamic masks or agent death. Keep the
        # public PettingZoo wrapper for specs/checks, avoiding its per-agent
        # conversions and copies on the vector collector's hot path.
        agents = env.possible_agents
        terminated = torch.tensor([terminated[a] for a in agents]).unsqueeze(-1)
        truncated = torch.tensor([truncated[a] for a in agents]).unsqueeze(-1)
        value = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.from_numpy(
                            np.stack([observations[a] for a in agents])
                        ),
                        "reward": torch.tensor(
                            [rewards[a] for a in agents], dtype=torch.float32
                        ).unsqueeze(-1),
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": terminated | truncated,
                    },
                    [len(agents)],
                ),
                "state": torch.from_numpy(env.state()),
                "terminated": terminated.any(0),
                "truncated": truncated.any(0),
                "done": (terminated | truncated).any(0),
            },
            [],
        )
        return self._metrics(value, env)

    def _metrics(self, td, env):
        values = torch.tensor(
            [
                env.config.delivery_load if k == "delivery_load" else env.last_metrics.get(k, 0)
                for k in self.metric_keys
            ],
            dtype=torch.float32,
        ).split(1)
        td["metrics"] = TensorDict(
            dict(zip(self.metric_keys, values, strict=True)),
            [],
        )
        return td

    def _reset(self, tensordict=None, **kwargs):
        mask = None if tensordict is None else tensordict.get("_reset", None)
        self._prepare_resets(
            [
                env
                for i, env in enumerate(self.envs)
                if mask is None or mask[i].any() or self.previous[i] is None
            ]
        )
        values = []
        for i, (env, wrapper) in enumerate(zip(self.envs, self.wrappers, strict=True)):
            if mask is None or mask[i].any() or self.previous[i] is None:
                value = self._metrics(wrapper.reset(), env)
            else:
                value = self.previous[i].exclude("reward", ("agents", "reward")).clone()
            values.append(value)
        self.previous = values
        return torch.stack(values)

    def _step(self, tensordict):
        targets = {}
        batch_actions = tensordict["agents", "action"].detach().cpu().numpy()
        for i, env in enumerate(self.envs):
            actions = batch_actions[i]
            env.begin(dict(zip(env.possible_agents, actions, strict=True)))
            targets[env.run_id] = i
        pending = set(targets)
        values = [None] * len(self.envs)
        while pending:
            responses = self.runner.recv_ready()
            received = perf_counter()
            for run_id, response in responses.items():
                if isinstance(response, SDKError):
                    raise response
                i = targets[run_id]
                env = self.envs[i]
                env.response = response
                env.response_received = received
                # Encode each ready worker while others are still simulating.
                # Keep worker slots/time axes stable for on-policy PPO/GAE.
                obs, rewards, terminated, truncated, _ = env.step(env.pending_actions)
                values[i] = self._pack_step(env, obs, rewards, terminated, truncated)
                pending.remove(run_id)
        self.previous = values
        return torch.stack(values)

    def _set_seed(self, seed):
        for env in self.envs:
            env.seed_value, env.generation = seed, 0
        return seed

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            for wrapper in self.wrappers:
                wrapper.close()
            self.runner.close()
            self.is_closed = True


class ContentTask(TaskClass):
    def __init__(
        self, config, episode_start=0, method="DEPPO-adapted", local_context=False, load_mix=()
    ):
        self.method = method
        self.local_context, self.load_mix = local_context, load_mix
        self.episode_start = episode_start
        super().__init__("content", config.model_dump())

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        if str(device) != "cpu":
            raise ValueError("SimGrid sampling must use CPU")
        return lambda: ContentBatchEnv(
            ScenarioConfig(**self.config),
            num_envs,
            seed,
            self.episode_start,
            self.method,
            self.local_context,
            self.load_mix,
        )

    def supports_continuous_actions(self):
        return True

    def supports_discrete_actions(self):
        return False

    def has_render(self, env):
        return False

    def max_steps(self, env):
        return self.config["cycles"]

    def group_map(self, env):
        return {"agents": [f"cluster-{i}" for i in range(self.config["clusters"])]}

    def observation_spec(self, env):
        spec = env.observation_spec[(0,) * len(env.batch_size)].clone()
        return Composite({"agents": spec["agents"]})

    def state_spec(self, env):
        spec = env.observation_spec[(0,) * len(env.batch_size)].clone()
        return Composite({"state": spec["state"]})

    def action_spec(self, env):
        return env.full_action_spec[(0,) * len(env.batch_size)].clone()

    def info_spec(self, env):
        return None

    def action_mask_spec(self, env):
        return None

    @staticmethod
    def env_name():
        return "edge_content"
