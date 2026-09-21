"""Local CPU workflow check: teacher -> independent distillation -> paired MAPPO."""

import argparse
import json
from pathlib import Path

import torch
from edge_sim_learning.experiment import evaluate, train
from edge_sim_learning.legacy_distill import actors_in, distilled_policy, fit
from edge_sim_learning.pretrain import collect, sha256
from edge_sim_learning.scenario import ScenarioConfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise RuntimeError("completed output exists; use a fresh output")
    cfg = ScenarioConfig.profile("small").model_copy(
        update={"scheduler_release": "window", "cycles": 16}
    )
    opts = dict(
        method="MAPPO-no-context",
        workers=2,
        frames_per_batch=64,
        minibatch=64,
        epochs=5,
        learning_rate=3e-4,
        normalize_advantage=True,
        hidden_size=256,
        context_size=128,
        initial_std=0.3,
        eval_interval=128,
        eval_episodes=2,
        eval_workers=2,
        eval_stochastic=True,
        mode="offline",
        device="cpu",
    )
    (out / "spec.json").write_text(
        json.dumps(
            dict(
                scenario=cfg.model_dump(),
                training=opts,
                teacher_seed=17,
                paired_seed=0,
                limitation="Workflow smoke only: shorter episodes/batches, no convergence claim",
                no_server_training=True,
            ),
            indent=2,
        )
    )
    print("TRAIN TEACHER", flush=True)
    train(cfg, out / "teacher", seed=17, episodes=16, **opts)
    teacher = out / "teacher/best-stochastic.pt"
    assert teacher.exists()
    print("COLLECT INDEPENDENT EPISODES", flush=True)
    manifest = collect(teacher, out / "data", train_episodes=8, validation_episodes=2, workers=2)
    print("DISTILL INDEPENDENT ACTORS, COMPLETE EPOCHS", flush=True)
    fit(manifest, out / "student", epochs=10, batch_size=32)
    student = fit(manifest, out / "student", epochs=20, batch_size=32, resume=True)
    teacher_payload = torch.load(teacher, map_location="cpu", weights_only=False)
    seeds = [1_000_000_000, 1_000_000_001]
    reports = {}
    print("CLOSED-LOOP PRETRAIN VALIDATION", flush=True)
    for name, policy in [
        ("teacher", teacher_payload["policy"]),
        ("frozen-student", distilled_policy(teacher_payload["policy"], student)),
    ]:
        reports[name] = evaluate(
            cfg, "MAPPO-no-context", policy, seeds, exploration="stochastic", workers=2
        )
    (out / "before-rl.json").write_text(json.dumps(reports, indent=2))
    # Proceed for workflow verification even if this tiny teacher/student is not useful.
    for arm in ["scratch", "pretrained"]:
        print("TRAIN", arm, flush=True)
        train(
            cfg,
            out / arm,
            seed=0,
            episodes=16,
            actor_init=student if arm == "pretrained" else None,
            **opts,
        )
        train(
            cfg, out / (arm + "-resumed"), seed=0, episodes=32, resume=out / arm / "last.pt", **opts
        )
        payload = torch.load(
            out / (arm + "-resumed/last.pt"), map_location="cpu", weights_only=False
        )
        assert payload["experiment"]["state"]["total_frames"] == 512
        reports[arm] = evaluate(
            cfg, "MAPPO-no-context", payload["policy"], seeds, exploration="stochastic", workers=2
        )
    initial = {
        arm: torch.load(out / arm / "initial.pt", weights_only=False)
        for arm in ("scratch", "pretrained")
    }
    student_payload = torch.load(student, weights_only=True)
    for actor, expected in zip(
        actors_in(initial["pretrained"]["policy"]), student_payload["actors"], strict=True
    ):
        for key, value in expected.items():
            assert torch.equal(actor.state_dict()[key], value)
    a, b = [initial[arm]["experiment"]["loss_agents"] for arm in ("scratch", "pretrained")]
    critic_keys = [k for k in a if "critic" in k and torch.is_tensor(a[k])]
    assert critic_keys and all(torch.equal(a[k], b[k]) for k in critic_keys)
    for arm, payload in initial.items():
        actors = actors_in(payload["policy"])
        assert len({actor.mlp[0].weight.data_ptr() for actor in actors}) == cfg.clusters
        for group in payload["optimizers"].values():
            for optimizer in group.values():
                assert not optimizer["state"], "RL must start with fresh optimizer state"
        final = torch.load(out / (arm + "-resumed/last.pt"), weights_only=False)
        for first, last in zip(actors, actors_in(final["policy"]), strict=True):
            assert not torch.equal(first.mlp[-1].weight[5:], last.mlp[-1].weight[5:])
    workloads = [[(e["seed"], e["workload"]) for e in r["episodes"]] for r in reports.values()]
    assert all(w == workloads[0] for w in workloads)
    history = torch.load(out / "student/last.pt", weights_only=True)["history"]
    report = dict(
        status="workflow-passed",
        performance_claim=False,
        teacher_sha256=sha256(teacher),
        student_sha256=sha256(student),
        distillation=history,
        evaluation={k: v["mean"] for k, v in reports.items()},
        checks=[
            "independent actors",
            "identical initial critics",
            "fresh RL optimizers",
            "learnable scales updated",
            "paired workloads",
            "BC and RL resume",
        ],
        student_success_gap_pp=100
        * (
            reports["frozen-student"]["mean"]["success_rate"]
            - reports["teacher"]["mean"]["success_rate"]
        ),
    )
    (out / "evaluations.json").write_text(json.dumps(reports, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
