"""Finite-value checks and per-agent KL; recovery uses the last completed batch."""

import json
from pathlib import Path

import torch
from benchmarl.experiment import Experiment


def require_finite(values, label):
    for value in values:
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {label}")


def actor_kl(loss, data, group):
    probe = data.clone()
    with torch.no_grad(), loss.actor_network_params.to_module(loss.actor_network):
        dist = loss.actor_network.get_dist(probe)
        require_finite([probe[group, "loc"], probe[group, "scale"]], "distribution")
        old = torch.distributions.Normal(data[group, "loc"], data[group, "scale"])
        new = torch.distributions.Normal(probe[group, "loc"], probe[group, "scale"])
        # Both policies use the identical invertible tanh/affine transform.
        kl = torch.distributions.kl_divergence(old, new).sum(-1)
        return kl.reshape(-1, kl.shape[-1]).mean(0), dist.log_prob(data[group, "action"])


class StableExperiment(Experiment):
    target_kl = 0.02

    def _optimizer_loop(self, group):
        data = self.replay_buffers[group].sample().to(self.config.train_device)
        loss = self.losses[group]
        optimizers = self.optimizers[group]
        if getattr(self, "_guard_frame", None) != self.total_frames:
            self._guard_frame = self.total_frames
            self._stopped_groups = set()
            self._actor_updates = 0
        try:
            kl, log_prob = actor_kl(loss, data, group)
            require_finite([kl, log_prob, data[group, "log_prob"]], "policy probabilities")
            # The first update of every rollout must reproduce behaviour log probabilities.
            if self._actor_updates == 0 and group not in self._stopped_groups:
                if not torch.allclose(log_prob, data[group, "log_prob"], atol=2e-4, rtol=2e-4):
                    raise FloatingPointError("rollout log probability recomputation mismatch")
            if kl.max() > self.target_kl:
                self._stopped_groups.add(group)
            values = loss(data)
            require_finite(values.values(), "loss")
            report = values.detach().clone()
            values = self.algorithm.process_loss_vals(group, values)
            for name, value in values.items():
                if name not in optimizers:
                    continue
                optimizer = optimizers[name]
                params = [p for pg in optimizer.param_groups for p in pg["params"]]
                update = name != "loss_objective" or group not in self._stopped_groups
                norm = 0.0
                optimizer.zero_grad(set_to_none=True)
                if update:
                    value.backward()
                    require_finite((p.grad for p in params if p.grad is not None), "gradient")
                    norm = self._grad_clip(optimizer)
                    require_finite([torch.as_tensor(norm)], "gradient norm")
                    optimizer.step()
                    require_finite(params, "updated parameters")
                    for state in optimizer.state.values():
                        require_finite(state.values(), "optimizer state")
                    if name == "loss_objective":
                        self._actor_updates += 1
                optimizer.zero_grad(set_to_none=True)
                report[f"grad_norm_{name}"] = torch.as_tensor(norm, device=self.config.train_device)
            post_kl, _ = actor_kl(loss, data, group)
            require_finite([post_kl], "updated KL")
            if post_kl.max() > self.target_kl:
                self._stopped_groups.add(group)
            for i, value in enumerate(post_kl):
                report[f"actor_{i}_kl"] = value
            report["actor_updates"] = torch.tensor(
                float(self._actor_updates), device=post_kl.device
            )
            report["kl_early_stop"] = torch.tensor(
                float(group in self._stopped_groups), device=post_kl.device
            )
            callback = self._on_train_step(data, group)
            if callback is not None:
                report.update(callback)
            return report
        except (FloatingPointError, ValueError) as error:
            diagnostic = {
                "error": str(error),
                "env_steps": self.total_frames,
                "resume_from": "last.pt",
            }
            (Path(self.config.save_folder) / "failure.json").write_text(
                json.dumps(diagnostic, indent=2)
            )
            raise FloatingPointError(
                f"PPO stopped; resume from the last healthy checkpoint: {error}"
            ) from error
