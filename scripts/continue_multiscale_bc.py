"""Extend supervised training on the same frozen data; no collection or RL."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from edge_sim_learning import multiscale_bc as bc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Previous continuation output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-at-end-only", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    interval = args.epochs if args.eval_at_end_only else 20
    if args.workers < 1:
        parser.error("workers must be positive")
    torch.set_num_threads(1)
    bc.EVAL_WORKERS = args.workers
    source, output = args.source.resolve(), args.output.resolve()
    phase = output / "continued"
    phase.mkdir(parents=True, exist_ok=True)
    records = json.loads((source / "round-0/data/manifest.json").read_text())
    previous = args.resume_from.resolve() if args.resume_from else source
    previous_phase = previous / "continued" if args.resume_from else source / "round-0"
    original = previous_phase / "last.pt"
    cfg = torch.load(original, map_location="cpu", weights_only=True)["config"]
    bc.freeze_spec(
        output / "spec.json",
        dict(
            source=str(source),
            initial_sha256=bc.sha256(original),
            data=[r["sha256"] for r in records],
            epochs=args.epochs,
            interval=interval,
            seed_split="validation",
            no_new_data=True,
        ),
    )
    if (output / "status.json").exists():
        print((output / "status.json").read_text())
        return
    initial = phase / "initial.pt"
    if not initial.exists():
        shutil.copyfile(original, initial)
    if bc.sha256(initial) != bc.sha256(original):
        raise ValueError("initial snapshot changed")
    if not (phase / "metrics.json").exists():
        shutil.copyfile(previous_phase / "metrics.json", phase / "metrics.json")
    source_spec = json.loads((source / "spec.json").read_text())
    conditions = {
        n: bc.ScenarioConfig.model_validate(c) for n, c in source_spec["conditions"].items()
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
    selection_path = output / "selection.json"
    if not selection_path.exists():
        shutil.copyfile(previous / "selection.json", selection_path)
    selected = json.loads(selection_path.read_text())
    best = output / "best.pt"
    if not best.exists() or bc.sha256(best) != selected["checkpoint_sha256"]:
        shutil.copyfile(selected["checkpoint"], best)

    def assess(checkpoint, tag):
        report = bc.compare_conditions(
            output, conditions, checkpoint, baselines, "validation", 4, tag
        )
        old = json.loads(selection_path.read_text())
        if tuple(report["rank"]) < tuple(old["rank"]):
            temporary = best.with_suffix(".tmp")
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(best)
            bc.write_json(
                selection_path,
                dict(report, checkpoint=str(checkpoint), checkpoint_sha256=bc.sha256(checkpoint)),
            )
        print(
            f"VALIDATE {tag} passed={sum(r['passed'] for r in report['rows'])}"
            f"/{len(conditions)} worst={report['rank'][0]:.3f}",
            flush=True,
        )
        return report

    bc.fit_phase(
        phase,
        records,
        initial,
        args.device,
        cfg["hidden_size"],
        args.epochs,
        cfg["batches"],
        cfg["batch_size"],
        interval,
        assess,
        continue_initial=True,
    )
    selected = json.loads(selection_path.read_text())
    bc.export_status(
        output,
        "validation-ready" if selected["passed"] else "base-not-ready",
        selected,
        "固定离线数据追加训练结束；仅开发验证，未运行独立测试或 RL。",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((phase / "metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ["train_nll", "validation_nll"]:
        ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
    ax.set(xlabel="Epoch", ylabel="Action NLL", title="BC continuation on fixed offline data")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
