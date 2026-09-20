"""Explicit history windows make shuffled PPO recomputation reproducible."""

from dataclasses import dataclass

import torch
from benchmarl.models import Model, ModelConfig
from torch import nn

from .env import ACTION, HISTORY, OBS


class HistoryActor(nn.Module):
    def __init__(self, output_dim=8, use_context=True):
        super().__init__()
        self.use_context = use_context
        if use_context:
            self.gru = nn.GRU(OBS + ACTION, 64, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(OBS + (64 if use_context else 0), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, features):
        current = features[..., :OBS]
        if self.use_context:
            history = features[..., OBS:-1].reshape(-1, HISTORY, OBS + ACTION)
            length = features[..., -1].reshape(-1).long().clamp(0, HISTORY)
            output, _ = self.gru(history)
            context = output[
                torch.arange(len(length), device=length.device), (length - 1).clamp_min(0)
            ]
            context = context * (length > 0).unsqueeze(-1)
            current = torch.cat((current, context.reshape(*features.shape[:-1], 64)), -1)
        return self.mlp(current)


class ContextModel(Model):
    def __init__(self, use_context=True, **kwargs):
        super().__init__(**kwargs)
        if self.centralised or self.share_params or not self.input_has_agent_dim:
            raise ValueError("DEPPO actor requires independent local agent parameters")
        self.actors = nn.ModuleList(
            [
                HistoryActor(self.output_leaf_spec.shape[-1], use_context)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)

    def _forward(self, tensordict):
        x = tensordict[self.in_key]
        tensordict[self.out_key] = torch.stack(
            [actor(x[..., i, :]) for i, actor in enumerate(self.actors)], -2
        )
        return tensordict


@dataclass
class ContextConfig(ModelConfig):
    use_context: bool = True

    @staticmethod
    def associated_class():
        return ContextModel
