import importlib.util
from pathlib import Path

import numpy as np
from edge_sim_learning.experiment import evaluate
from edge_sim_learning.scenario import ScenarioConfig, build_run


def test_fixed_capacity_legacy_sweep():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "legacy_loads", root / "scripts/evaluate_legacy_loads.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ScenarioConfig.model_validate_json(
        (root / "experiments/legacy-mappo-f064/scenario.json").read_text()
    )
    reference, _ = build_run(base, 42)
    for load in [0.25, 0.5, 0.75, 1, 1.25]:
        cfg = module.load_config(base, load)
        run, _ = build_run(cfg, 42)
        assert not cfg.max_retries and not cfg.episode_loads
        assert cfg.request_rate == 400 * load
        assert np.allclose(
            [c.total_bandwidth_bytes_s for c in reference.content.caches],
            [c.total_bandwidth_bytes_s for c in run.content.caches],
            rtol=1e-12,
        )
    cfg = module.load_config(base, 0.25).model_copy(update={"cycles": 2})
    result = evaluate(cfg, "local", seeds=[1_000_000_000], workers=1)
    assert result["mean"]["unfinished"] == 0
