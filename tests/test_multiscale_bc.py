"""Local context, executable supervision, resume and per-condition acceptance."""

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch
from edge_sim_learning.local_context import CONTEXT_DIM, LocalContextEnv
from edge_sim_learning.multiscale_bc import (
    BASELINES,
    SEED_BASES,
    TEACHERS,
    BalancedData,
    ContextActor,
    _collect_one,
    collect_jobs,
    distribution,
    fit_phase,
    gate,
    make_conditions,
    paired_ci,
    policy_from_payload,
    safe_actions,
    seeds,
    teacher_actions,
)
from edge_sim_learning.pretrain import load_base_policy
from edge_sim_models import ContentArrivalState


def config():
    return next(iter(make_conditions(None, smoke=True).values()))


def test_local_context_telemetry_and_isolation():
    env = LocalContextEnv(config())
    try:
        obs, _ = env.reset(seed=7)
        a, b = env.possible_agents
        assert obs[a].shape == (CONTEXT_DIM,)
        assert np.all(obs[a][-4:] == 0)
        arrival = ContentArrivalState(request_id="x", cluster_id=a, arrival_s=0.25, size_bytes=100)
        view = env.last_view.model_copy(update={"now_s": 0.5, "window_arrivals": (arrival,)})
        env._encode(view)
        env.last_view = view
        changed = env._observations()
        capacity = env.resources[a][0]
        assert changed[a][17] == pytest.approx(200 / capacity)
        assert changed[b][17] == 0
        # No double count on repeated snapshots; expire only actual past arrivals.
        env._encode(view)
        assert len(env.arrival_history[a]) == 1
        env._encode(view.model_copy(update={"now_s": 1.26, "window_arrivals": ()}))
        assert not env.arrival_history[a]
        reset, _ = env.reset(seed=7)
        assert all(v[-1] == 0 for v in reset.values())
        # Real telemetry includes retries, reconciles the attempt-level arrival count.
        total = 0
        for _ in range(env.config.cycles):
            env.step(teacher_actions(env, "local-0.5"))
            total += len(env.last_view.window_arrivals)
        assert total == env.last_view.arrived
    finally:
        env.close()


def test_teachers_are_five_dimensional_and_state_local():
    env = LocalContextEnv(config())
    try:
        env.reset(seed=8)
        for teacher in TEACHERS:
            actions = teacher_actions(env, teacher)
            control = env.control(actions)
            assert control.policy == "threshold"
            for item in control.schedulers:
                assert item.weights[:3] == (0, 0, 0)
                assert item.weights[3] == (-2 if teacher.startswith("local-") else 2)
                assert 0.05 <= item.backhaul_ratio <= 0.95
        assert len(BASELINES) == 5
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_finite_v2_distribution_and_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires NVIDIA GPU")
    torch.manual_seed(0)
    actor = ContextActor(32).to(device)
    obs = torch.randn(128, CONTEXT_DIM, device=device)
    labels = torch.tensor([0, 0, 0, 2, 0.5], device=device).repeat(128, 1)
    opt = torch.optim.Adam(actor.parameters(), lr=0.003)
    initial = -distribution(actor, obs)[0].log_prob(labels).mean().item()
    for _ in range(30):
        loss = -distribution(actor, obs)[0].log_prob(labels).mean()
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
        opt.step()
    assert loss.item() < initial
    bounds = torch.tensor([[-5, -5, -5, -5, 0.05], [5, 5, 5, 5, 0.95]], device=device)
    assert torch.isfinite(distribution(actor, obs[:2])[0].log_prob(safe_actions(bounds))).all()
    with pytest.raises(ValueError, match="21"):
        actor(torch.zeros(2, 13))


def test_acceptance_and_paired_workload():
    def report(success, latency):
        m = dict(logical_success_rate=success, mean_success_e2e_s=latency)
        return dict(mean=m, episodes=[dict(m, seed=i, workload={"seed": i}) for i in range(4)])

    ref = report(0.8, 1.0)
    assert gate(report(0.79, 1.05), ref)["passed"]
    assert not gate(report(0.789, 0.1), ref)["passed"]
    assert not gate(report(0.99, 1.051), ref)["passed"]
    ci = paired_ci(report(0.81, 0.9), ref)
    assert ci["success_difference_pp_ci95"] == pytest.approx([1, 1])
    seen = set()
    for split in SEED_BASES:
        for condition in range(15):
            for r in range(3):
                new = set(seeds(split, condition, 32, r))
                assert not new & seen
                seen |= new


def test_real_collection_checkpoint_resume_and_dagger(tmp_path):
    cfg = config()
    jobs = []
    for split, seed in [("train", 1), ("supervised-validation", 2)]:
        jobs.append(
            (
                cfg.model_dump(),
                "local-0.5",
                str(tmp_path / f"{split}.pt"),
                seed,
                split,
                "smoke",
                None,
            )
        )
    records = collect_jobs(tmp_path, jobs, 1)
    assert records == collect_jobs(tmp_path, jobs, 1)
    d = torch.load(records[0]["file"], weights_only=True)
    assert torch.equal(d["action"], d["teacher_action"])
    assert torch.equal(d["next_observation"][:-1], d["observation"][1:])
    assert d["truncated"].sum() == 1 and not d["terminated"].any()
    train = BalancedData(records, "train")
    assert train.sample(16, torch.Generator())[0].shape == (16, 21)
    with pytest.raises(ValueError, match="duplicate"):
        BalancedData(records + records, "train")
    calls = []

    def assess(path, tag):
        calls.append(tag)
        return {"rows": []}

    folder = tmp_path / "fit"
    fit_phase(folder, records, None, "cpu", 32, 2, 2, 16, 1, assess)
    before = (folder / "last.pt").read_bytes()
    fit_phase(folder, records, None, "cpu", 32, 2, 2, 16, 1, assess)
    assert (folder / "last.pt").read_bytes() == before
    base = load_base_policy(folder / "last.pt")
    assert base.observation_profile == "local-context-v2"
    payload = torch.load(folder / "last.pt", weights_only=True)
    assert policy_from_payload(payload).observation_profile == base.observation_profile
    row = _collect_one(
        (
            cfg.model_dump(),
            "local-0.5",
            str(tmp_path / "dagger.pt"),
            9,
            "train",
            "smoke",
            str(folder / "last.pt"),
        )
    )
    dagger = torch.load(row["file"], weights_only=True)
    assert not torch.equal(dagger["action"], dagger["teacher_action"])
    assert dagger["behavior_sha256"]
    assert dagger["metrics"]["logical_unfinished"] == 0


def test_actual_retry_arrivals_are_counted():
    cfg = config().model_copy(update={"deadline_s": 0.01, "request_rate": 100})
    env = LocalContextEnv(cfg)
    arrivals = []
    original = env._encode

    def capture(view):
        arrivals.extend(view.window_arrivals)
        original(view)

    env._encode = capture
    try:
        env.reset(seed=19)
        while env.agents:
            env.step(teacher_actions(env, "local-0.5"))
        result = env.drain(lambda _: teacher_actions(env, "local-0.5"))
        assert result["retry_attempts"] > 0
        assert len(arrivals) == result["arrived"]
        assert len({a.request_id for a in arrivals}) == len(arrivals)
        assert (
            sum(a.request_id.startswith("__retry_") for a in arrivals) == result["retry_attempts"]
        )
    finally:
        env.close()


@pytest.mark.parametrize("teacher", TEACHERS)
def test_collector_teacher_matches_baseline_execution(tmp_path, teacher):
    from edge_sim_learning.experiment import evaluate
    from edge_sim_learning.multiscale_bc import teacher_method

    cfg = config()
    row = _collect_one(
        (cfg.model_dump(), teacher, str(tmp_path / "episode.pt"), 77, "train", "smoke", None)
    )
    collected = torch.load(row["file"], weights_only=True)
    method, ratio = teacher_method(teacher)
    result = evaluate(cfg, method, seeds=[77], fixed_ratio=ratio, exploration="stochastic")
    for key in [
        "logical_success_rate",
        "mean_success_e2e_s",
        "mean_failed_elapsed_s",
        "arrived",
        "backhaul_bytes",
        "delivery_bytes",
    ]:
        assert collected["metrics"][key] == pytest.approx(result["mean"][key])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_interrupted_evaluation_resumes_exact_training_state(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires NVIDIA GPU")
    cfg = config()
    records = collect_jobs(
        tmp_path,
        [
            (
                cfg.model_dump(),
                "local-0.5",
                str(tmp_path / f"{split}.pt"),
                seed,
                split,
                "smoke",
                None,
            )
            for split, seed in [("train", 22), ("supervised-validation", 23)]
        ],
        1,
    )

    def ok(path, tag):
        return {"rows": []}

    def interrupt(path, tag):
        raise RuntimeError("simulated evaluator interruption")

    args = (records, None, device, 32, 2, 2, 16, 1)
    fit_phase(tmp_path / "reference", *args, ok)
    with pytest.raises(RuntimeError, match="simulated"):
        fit_phase(tmp_path / "resumed", *args, interrupt)
    assert torch.load(tmp_path / "resumed/last.pt", weights_only=True)["epoch"] == 1
    fit_phase(tmp_path / "resumed", *args, ok)
    a = torch.load(tmp_path / "reference/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert all(torch.equal(a["actor"][key], b["actor"][key]) for key in a["actor"])


def test_parallel_conditions_match_serial(tmp_path):
    import edge_sim_learning.multiscale_bc as bc

    cfg = config()
    old = bc.EVAL_WORKERS
    try:
        bc.EVAL_WORKERS = 8
        parallel = bc.evaluation_batch(
            [
                (tmp_path / f"parallel-{i}", cfg.model_dump(), [101, 102], method, None, None)
                for i, method in enumerate(["local", "forward"])
            ]
        )
        bc.EVAL_WORKERS = 1
        serial = bc.evaluation_batch(
            [
                (tmp_path / f"serial-{i}", cfg.model_dump(), [101, 102], method, None, None)
                for i, method in enumerate(["local", "forward"])
            ]
        )
        for a, b in zip(parallel, serial, strict=True):
            for key in ["logical_success_rate", "mean_success_e2e_s", "mean_failed_elapsed_s"]:
                assert a["mean"][key] == pytest.approx(b["mean"][key])
    finally:
        bc.EVAL_WORKERS = old
