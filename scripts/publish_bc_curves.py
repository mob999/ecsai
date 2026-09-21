"""Publish completed fixed-data BC metrics as W&B charts, without rerunning simulation."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    metrics, reports = {}, {}
    for root in args.roots:
        for path in root.glob("*/metrics.json"):
            for row in json.loads(path.read_text()):
                metrics[row["epoch"]] = row
        for path in root.glob("evaluations/*/comparison.json"):
            epoch = int(path.parent.name.rsplit("-", 1)[1])
            reports[epoch] = json.loads(path.read_text())
    with wandb.init(
        project="ecsai-deppo",
        name="BC-original-1-to-200-summary",
        mode="online",
        config={
            "sources": [str(p) for p in args.roots],
            "evaluation": "stochastic, fixed validation seeds, retries included",
        },
    ) as run:
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        for epoch, row in sorted(metrics.items()):
            values = dict(row)
            if epoch in reports:
                rows = reports[epoch]["rows"]
                values["eval/passed_conditions"] = sum(r["passed"] for r in rows)
                values["eval/mean_success_rate"] = sum(
                    r["logical_success_rate"] for r in rows
                ) / len(rows)
                for r in rows:
                    for key in [
                        "logical_success_rate",
                        "mean_success_e2e_s",
                        "mean_failed_elapsed_s",
                        "mean_resolution_time_s",
                        "mean_attempts",
                        "success_difference_pp",
                        "latency_ratio",
                    ]:
                        values[f"eval/{r['condition']}/{key}"] = r[key]
            run.log(values)
        epochs = sorted(metrics)
        charts = {
            "charts/loss": wandb.plot.line_series(
                xs=epochs,
                ys=[[metrics[e][k] for e in epochs] for k in ["train_nll", "validation_nll"]],
                keys=["Training NLL", "Validation NLL"],
                title="BC action NLL (original model)",
                xname="Epoch",
            )
        }
        eval_epochs = sorted(reports)
        names = [r["condition"] for r in reports[eval_epochs[0]]["rows"]]
        for key, title in [
            ("logical_success_rate", "Final request success rate"),
            ("mean_success_e2e_s", "Successful request total latency incl. retries (s)"),
            ("mean_resolution_time_s", "All request resolution time (s)"),
            ("success_difference_pp", "Success difference vs best baseline (percentage points)"),
        ]:
            charts[f"charts/{key}"] = wandb.plot.line_series(
                xs=eval_epochs,
                ys=[
                    [
                        next(r[key] for r in reports[e]["rows"] if r["condition"] == name)
                        for e in eval_epochs
                    ]
                    for name in names
                ],
                keys=names,
                title=title,
                xname="Epoch",
            )
        charts["charts/passed_conditions"] = wandb.plot.line_series(
            xs=eval_epochs,
            ys=[[sum(r["passed"] for r in reports[e]["rows"]) for e in eval_epochs]],
            keys=["Passed / 15"],
            title="Validation conditions passing both gates",
            xname="Epoch",
        )
        run.log(charts)
        print("DASHBOARD_URL", run.url, flush=True)


if __name__ == "__main__":
    main()
