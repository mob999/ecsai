"""Vector sampling over SDK workers; no second layer of process vectorization."""

import torch
from benchmarl.environments.common import TaskClass
from edge_sim import BatchRunner, SDKError
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, PettingZooWrapper

from .env import SchedulingEnv
from .scenario import ScenarioConfig


class ContentBatchEnv(EnvBase):
    def __init__(self, config, workers=4, seed=0, episode_start=0):
        super().__init__(device="cpu", batch_size=[workers])
        self.runner = BatchRunner(workers=workers)
        self.envs = [
            SchedulingEnv(config, seed=seed, runner=self.runner, slot=i) for i in range(workers)
        ]
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
                "episode_return",
                "simulation_wall_s",
                "ipc_wall_s",
                "window_wall_s",
                "window_completed",
                "window_resolved",
            }
        )
        observation = prototype.observation_spec.clone()
        observation["metrics"] = Composite({k: Unbounded(shape=(1,)) for k in self.metric_keys})
        self.observation_spec = observation.expand(workers)
        self.action_spec = prototype.full_action_spec.expand(workers)
        self.reward_spec = prototype.full_reward_spec.expand(workers)
        self.done_spec = prototype.full_done_spec.expand(workers)
        self.previous = [None] * workers
        self.is_closed = False

    def _metrics(self, td, env):
        td["metrics"] = TensorDict(
            {
                k: torch.tensor([env.last_metrics.get(k, 0)], dtype=torch.float32)
                for k in self.metric_keys
            },
            [],
        )
        return td

    def _reset(self, tensordict=None, **kwargs):
        mask = None if tensordict is None else tensordict.get("_reset", None)
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
        for i, env in enumerate(self.envs):
            actions = tensordict["agents", "action"][i].detach().cpu().numpy()
            env.begin(dict(zip(env.possible_agents, actions, strict=True)))
            targets[env.run_id] = env
        pending = set(targets)
        while pending:
            for run_id, response in self.runner.recv_ready().items():
                if isinstance(response, SDKError):
                    raise response
                targets[run_id].response = response
                pending.remove(run_id)
        values = [
            self._metrics(wrapper.step(tensordict[i])["next"], env)
            for i, (env, wrapper) in enumerate(zip(self.envs, self.wrappers, strict=True))
        ]
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
    def __init__(self, config, episode_start=0):
        self.episode_start = episode_start
        super().__init__("content", config.model_dump())

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        if str(device) != "cpu":
            raise ValueError("SimGrid sampling must use CPU")
        return lambda: ContentBatchEnv(
            ScenarioConfig(**self.config), num_envs, seed, self.episode_start
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
