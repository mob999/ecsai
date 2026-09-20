"""Measure steady-state and end-to-end sampling separately, with real workers."""

from time import perf_counter

import torch

from .experiment import metadata
from .torch_env import ContentBatchEnv


def benchmark(config, workers=(1, 2, 4), steps=128, seed=0):
    torch.set_num_threads(1)
    rows = []
    if steps < 1 or not workers or any(count < 1 for count in workers):
        raise ValueError("positive steps and workers required")
    for count in workers:
        torch.manual_seed(seed)
        started = perf_counter()
        env = ContentBatchEnv(config, count, seed)
        try:
            td = env.reset()
            setup = perf_counter() - started
            sampling = perf_counter()
            native = ipc = processed = 0.0
            for _ in range(steps):
                td = env.rand_action(td)
                next_td = env.step(td)["next"]
                metrics = next_td["metrics"]
                native += metrics["simulation_wall_s"].sum().item()
                ipc += metrics["ipc_wall_s"].sum().item()
                processed += metrics["window_resolved"].sum().item()
                if next_td["done"].any():
                    next_td["_reset"] = next_td["done"]
                    td = env.reset(next_td)
                else:
                    td = next_td.exclude("reward", ("agents", "reward"))
            wall = perf_counter() - sampling
            rows.append(
                {
                    "workers": count,
                    "env_steps": steps * count,
                    "sampling_wall_s": wall,
                    "setup_wall_s": setup,
                    "steps_s": steps * count / wall,
                    "requests_s": processed / wall,
                    "simulation_worker_seconds": native,
                    "ipc_and_wait_worker_seconds": ipc,
                }
            )
        finally:
            env.close()
    return {"metadata": metadata(config, seed, "random"), "results": rows}
