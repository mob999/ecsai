"""Explicit history windows make shuffled PPO recomputation reproducible."""

import math
from dataclasses import dataclass

import torch
from benchmarl.models import Model, ModelConfig
from torch import nn

from .env import ACTION, HISTORY, OBS


class HistoryActor(nn.Module):
    def __init__(
        self,
        output_dim=2 * ACTION,
        use_context=True,
        hidden_size=128,
        context_size=64,
        initial_std=None,
        input_dim=OBS,
    ):
        super().__init__()
        self.use_context = use_context
        self.input_dim = input_dim
        if use_context:
            self.gru = nn.GRU(OBS + ACTION, context_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + (context_size if use_context else 0), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

        if initial_std is not None:
            if not 0.1 <= initial_std <= 1.5:
                raise ValueError("initial_std must be in [0.1, 1.5]")
            # Invert BenchMARL biased_softplus_1.0 (floor .01); loc init is unchanged.
            raw = math.log(math.expm1(initial_std - 0.01)) - math.log(math.expm1(0.99))
            with torch.no_grad():
                self.mlp[-1].weight[output_dim // 2 :].zero_()
                self.mlp[-1].bias[output_dim // 2 :].fill_(raw)

    def forward(self, features):
        input_dim = getattr(self, "input_dim", OBS)
        current = features[..., :input_dim]
        if self.use_context:
            history = features[..., OBS:-1].reshape(-1, HISTORY, OBS + ACTION)
            length = features[..., -1].reshape(-1).long().clamp(0, HISTORY)
            output, _ = self.gru(history)
            context = output[
                torch.arange(len(length), device=length.device), (length - 1).clamp_min(0)
            ]
            context = context * (length > 0).unsqueeze(-1)
            current = torch.cat(
                (current, context.reshape(*features.shape[:-1], self.gru.hidden_size)), -1
            )
        raw = self.mlp(current)
        if input_dim != OBS:
            slots = features[..., OBS:].reshape(*features.shape[:-1], -1, 4)
            valid = (slots[..., 3] > 0) & (slots[..., 2] == 0)
            mask = torch.cat((valid, torch.ones_like(valid[..., :1])), -1)
            # Padding/forwarded slots have a fixed distribution: zero policy gradient
            # and zero KL, with identical factors cancelling in PPO likelihood ratios.
            raw = torch.where(torch.cat((mask, mask), -1), raw, torch.zeros_like(raw))
        loc, raw_scale = raw.chunk(2, dim=-1)
        # BenchMARL applies biased_softplus_1.0 to the second half.
        # Bound pre-tanh means and keep standard deviations in a finite useful range.
        return torch.cat((3 * torch.tanh(loc / 3), raw_scale.clamp(-3, 1)), -1)


class ContextModel(Model):
    def __init__(
        self,
        use_context=True,
        hidden_size=128,
        context_size=64,
        initial_std=None,
        input_dim=OBS,
        actor_init=None,
        head_only=False,
        local_context=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.centralised or self.share_params or not self.input_has_agent_dim:
            raise ValueError("DEPPO actor requires independent local agent parameters")
        if local_context:
            from .multiscale_bc import actor_from_config

            base_config = dict(
                hidden_size=256,
                architecture=dict(depth=4, layer_norm=True, dropout=0.05),
                fixed_scale=0.1,
            )
            if (
                use_context
                or input_dim != 21
                or self.output_leaf_spec.shape[-1] != 10
                or hidden_size != 256
                or head_only
            ):
                raise ValueError("local-context RL requires the selected 4x256 five-action actor")
            self.actors = nn.ModuleList(
                [actor_from_config(base_config) for _ in range(self.n_agents)]
            ).to(self.device)
            if actor_init is not None:
                payload = torch.load(actor_init, map_location=self.device, weights_only=True)
                for key, value in base_config.items():
                    if payload["config"].get(key) != value:
                        raise ValueError(f"base actor mismatch: {key}")
                if (
                    payload["config"]["format"] != "edge-bc-v2"
                    or payload["config"]["input_dim"] != 21
                ):
                    raise ValueError("requires local-context-v2 checkpoint")
                for actor in self.actors:
                    actor.load_state_dict(payload["actor"], strict=True)
            # PPO recomputes likelihoods: disable dropout in both experimental arms.
            for actor in self.actors:
                for i, layer in enumerate(actor.mlp):
                    if isinstance(layer, nn.Dropout):
                        actor.mlp[i] = nn.Identity()
            return
        self.actors = nn.ModuleList(
            [
                HistoryActor(
                    self.output_leaf_spec.shape[-1],
                    use_context,
                    hidden_size,
                    context_size,
                    initial_std,
                    input_dim,
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        if actor_init is not None:
            payload = torch.load(actor_init, map_location=self.device, weights_only=True)
            config = payload["config"]
            if (
                config["format"] not in {"edge-bc-v1", "edge-distill-independent-v1"}
                or config["hidden_size"] != hidden_size
                or config["input_dim"] != input_dim
                or config["action_dim"] * 2 != self.output_leaf_spec.shape[-1]
                or use_context
            ):
                raise ValueError("base actor architecture mismatch")
            if config["format"] == "edge-distill-independent-v1":
                if config["agents"] != self.n_agents or len(payload["actors"]) != self.n_agents:
                    raise ValueError("independent distillation requires the original agent count")
                for actor, state in zip(self.actors, payload["actors"], strict=True):
                    actor.load_state_dict(state, strict=True)
            else:
                for actor in self.actors:
                    actor.load_state_dict(payload["actor"], strict=True)
        if head_only:
            if use_context or input_dim != OBS:
                raise ValueError("head-only adaptation requires a no-context actor")
            for actor in self.actors:
                for parameter in actor.parameters():
                    parameter.requires_grad_(False)
                for parameter in actor.mlp[-1].parameters():
                    parameter.requires_grad_(True)

    def _forward(self, tensordict):
        x = tensordict[self.in_key]
        tensordict[self.out_key] = torch.stack(
            [actor(x[..., i, :]) for i, actor in enumerate(self.actors)], -2
        )
        return tensordict


@dataclass
class ContextConfig(ModelConfig):
    use_context: bool = True
    hidden_size: int = 128
    context_size: int = 64
    initial_std: float | None = None
    input_dim: int = OBS
    actor_init: str | None = None
    head_only: bool = False
    local_context: bool = False

    @staticmethod
    def associated_class():
        return ContextModel
