"""Controlled fixed-data BC screening followed by paired four-episode confirmation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from edge_sim_learning import multiscale_bc as bc


def score(report):
    rows = report["rows"]
    return (
        sum(max(0, -r["success_difference_pp"]) for r in rows) / len(rows),
        max(max(0, -r["success_difference_pp"]) for r in rows),
        sum(r["latency_ratio"] for r in rows) / len(rows),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    variants = [
        (model + "-" + variant, model, flags)
        for model in ["original", "deep-reg"]
        for variant, flags in [
            ("nll", []),
            ("low-lr", ["--learning-rate", "0.0001"]),
            ("fixed-nll", ["--fixed-scale", "0.1"]),
            ("fixed-mse", ["--fixed-scale", "0.1", "--loss", "mean-mse"]),
        ]
    ]
    bc.freeze_spec(
        output / "search-spec.json",
        dict(
            variants=variants,
            epochs=20,
            screen_episodes=1,
            confirm_episodes=4,
            ranking="mean positive success deficit pp, worst deficit, mean latency ratio",
            source=str(source),
            no_rl=True,
        ),
    )
    screens = {}
    for name, model, flags in variants:
        dest = output / name
        cmd = [
            sys.executable,
            "scripts/continue_multiscale_bc.py",
            "--source",
            str(source),
            "--output",
            str(dest),
            "--epochs",
            "20",
            "--fresh",
            "--full-epoch",
            "--eval-at-end-only",
            "--eval-episodes",
            "1",
            "--workers",
            "12",
            "--device",
            "cuda",
        ]
        if model == "deep-reg":
            cmd.append("--fresh-regularized")
        cmd += flags
        with (output / f"{name}.log").open("a") as log:
            subprocess.run(
                cmd,
                env=dict(os.environ, WANDB_MODE="online", WANDB_NAME="BC-search-" + name),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report = json.loads((dest / "evaluations/continued-epoch-20/comparison.json").read_text())
        screens[name] = dict(score=score(report), checkpoint=str(dest / "continued/last.pt"))
        bc.write_json(output / "screening.json", screens)
        print("SCREEN", name, screens[name]["score"], flush=True)
    conditions = {
        n: bc.ScenarioConfig.model_validate(c)
        for n, c in json.loads((source / "spec.json").read_text())["conditions"].items()
    }
    baselines = {
        n: {
            label: json.loads(
                (source / "baselines-validation" / n / label / "evaluation.json").read_text()
            )
            for label in bc.BASELINES
        }
        for n in conditions
    }
    bc.EVAL_WORKERS = 12
    confirmed = {}
    # Confirm the best two plus the unchanged original NLL control.
    finalists = list(
        dict.fromkeys(sorted(screens, key=lambda n: screens[n]["score"])[:2] + ["original-nll"])
    )
    for name in finalists:
        report = bc.compare_conditions(
            output,
            conditions,
            screens[name]["checkpoint"],
            baselines,
            "validation",
            4,
            "confirm-" + name,
        )
        confirmed[name] = dict(
            score=score(report), checkpoint=screens[name]["checkpoint"], report=report
        )
        bc.write_json(output / "confirmation.json", confirmed)
        print("CONFIRM", name, confirmed[name]["score"], flush=True)
    winner = min(confirmed, key=lambda n: confirmed[n]["score"])
    result = dict(
        winner=winner,
        finalists=finalists,
        score=confirmed[winner]["score"],
        checkpoint=confirmed[winner]["checkpoint"],
        note="development validation only; no independent test or RL",
    )
    import wandb

    with wandb.init(
        project="ecsai-deppo", name="BC-controlled-search-summary", mode="online", dir=str(output)
    ) as run:
        run.log(
            {
                "confirmation": wandb.Table(
                    columns=[
                        "variant",
                        "mean_success_deficit_pp",
                        "worst_success_deficit_pp",
                        "mean_latency_ratio",
                    ],
                    data=[[n, *r["score"]] for n, r in confirmed.items()],
                )
            }
        )
        run.summary.update(result)
        result["wandb_url"] = run.url
    bc.write_json(output / "result.json", result)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
