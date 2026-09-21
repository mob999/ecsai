# ruff: noqa: E501
"""Render the completed BC search's local evidence and per-condition comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    p = args.folder
    result = json.loads((p / "result.json").read_text())
    confirmed = json.loads((p / "confirmation.json").read_text())
    screening = json.loads((p / "screening.json").read_text())
    winner = result["winner"]
    rows = confirmed[winner]["report"]["rows"]
    assert len(rows) == 15 and len(screening) == 8
    lines = [
        "# BC controlled search results",
        "",
        f"Recommended candidate: **{winner}**.",
        "",
        "Eight seed-0 runs, identical frozen data and 6,000 optimizer updates. "
        "One paired episode per condition for screening; four paired development episodes "
        "per condition for confirmation. This is not an independent test or proof of statistical superiority.",
        "",
        "Ranking minimizes mean positive success deficit (improvements do not cancel deficits), "
        "then worst success deficit, then mean latency ratio. All latency metrics include actual retries.",
        "",
        "| Confirmed variant | Mean success | Mean deficit / pp | Worst deficit / pp | Success latency / s | All-request elapsed / s | Failed elapsed / s | Mean attempts | Gates passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in sorted(confirmed.items(), key=lambda item: item[1]["score"]):
        r = v["report"]["rows"]
        avg = {
            k: sum(x[k] for x in r) / len(r)
            for k in (
                "logical_success_rate",
                "mean_success_e2e_s",
                "mean_resolution_time_s",
                "mean_failed_elapsed_s",
                "mean_attempts",
            )
        }
        lines.append(
            f"| {n} | {avg['logical_success_rate'] * 100:.2f}% | {v['score'][0]:.3f} | {v['score'][1]:.3f} | {avg['mean_success_e2e_s']:.3f} | {avg['mean_resolution_time_s']:.3f} | {avg['mean_failed_elapsed_s']:.3f} | {avg['mean_attempts']:.3f} | {sum(x['passed'] for x in r)}/15 |"
        )
    lines += [
        "",
        "All averages above give equal weight to conditions; failed-only averages use zero when no failures occur.",
        "",
        "## Recommended candidate by condition",
        "",
        "| Condition | Reference policy | Reference success | Candidate success | Difference / pp | Paired 95% interval / pp | Success latency / s | Latency ratio |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        ci = r["success_difference_pp_ci95"]
        lines.append(
            f"| {r['condition']} | {r['reference']} | {100 * r['reference_success_rate']:.2f}% | {100 * r['logical_success_rate']:.2f}% | {r['success_difference_pp']:+.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {r['mean_success_e2e_s']:.3f} | {r['latency_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Screening (one episode; not confirmation)",
        "",
        "| Variant | Mean deficit / pp | Worst deficit / pp |",
        "|---|---:|---:|",
    ]
    for n, v in sorted(screening.items(), key=lambda item: item[1]["score"]):
        lines.append(f"| {n} | {v['score'][0]:.3f} | {v['score'][1]:.3f} |")
    lines += [
        "",
        f"Checkpoint on server: `{result['checkpoint']}`.",
        "",
        f"W&B: {result.get('wandb_url', 'pending')}",
    ]
    (p / "report.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for col, size in enumerate(["small", "medium", "large"]):
        reference = [r for r in rows if r["condition"].startswith(size + "-")]
        loads = [float(r["condition"].split("rho")[1]) for r in reference]
        axes[0, col].plot(
            loads,
            [100 * r["reference_success_rate"] for r in reference],
            "--",
            color="black",
            label="Best baseline per condition",
        )
        axes[1, col].plot(
            loads, [r["reference_success_e2e_s"] for r in reference], "--", color="black"
        )
        for n, v in confirmed.items():
            r = [x for x in v["report"]["rows"] if x["condition"].startswith(size + "-")]
            axes[0, col].plot(
                loads, [100 * x["logical_success_rate"] for x in r], marker="o", label=n
            )
            axes[1, col].plot(loads, [x["mean_success_e2e_s"] for x in r], marker="o")
        axes[0, col].set_title(size)
        axes[1, col].set_xlabel("Delivery load ratio")
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Final request success (%)")
    axes[1, 0].set_ylabel("Successful request total latency (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(p / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
