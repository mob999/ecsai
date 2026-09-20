"""Configure reproducible W&B scale dashboards with fixed-evaluation references.

Run with isolated dependency wandb-workspaces==0.4.11; --dry-run requires no account.
"""

import argparse
import json
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

METRICS = {
    "episode_return": "验证累计 reward（随机执行）",
    "success_rate": "请求成功率",
    "mean_latency_s": "成功请求平均延迟（秒）",
    "latency_p95_s": "成功请求 P95 延迟（秒）",
    "timeout_rate": "超时率",
    "rejection_rate": "拒绝率",
    "backhaul_utilization": "回源带宽利用率",
    "delivery_utilization": "交付带宽利用率",
}


def build_workspace(entity, project, scale, baseline_ids):
    sizes = {"small": "3 clusters - 10 caches", "medium": "5 clusters - 20 caches", "large": "7 clusters - 30 caches"}
    panels = []
    for metric, title in METRICS.items():
        key = "eval_stochastic/" + metric
        panels.append(
            wr.LinePlot(
                title=title,
                x="env_steps",
                y=[key],
                title_x="训练环境步数",
                range_x=(0, 262144),
                smoothing_type="none",
                max_runs_to_show=7,
                line_marks={f"{rid}:{key}": "dashed" for rid in baseline_ids},
            )
        )
    return ws.Workspace(
        entity=entity,
        project=project,
        name=f"Scale comparison - {sizes[scale]} - rho 0.75",
        runset_settings=ws.RunsetSettings(filters=f"Group = 'window-scale-{scale}'"),
        sections=[
            ws.Section(
                name="固定验证集对比 · 虚线为基线评估参考值（非训练轨迹）",
                panels=panels,
                is_open=True,
                pinned=True,
                layout_settings=ws.SectionLayoutSettings(columns=2, rows=4),
            ),
            ws.Section(
                name="训练与执行方式诊断",
                is_open=True,
                panels=[
                    wr.LinePlot(title=title, x="env_steps", y=[key], smoothing_type="none")
                    for title, key in [
                        ("训练业务 reward", "train/business_reward"),
                        ("确定性执行成功率", "eval/success_rate"),
                        ("采样吞吐（环境步/秒）", "train/sampling_steps_s"),
                        ("采样耗时（秒）", "train/collection_wall_s"),
                    ]
                ],
            ),
        ],
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/scale-matrix-v1"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    entity, project = "gc-beijing-jiaotong-university", "ecsai-deppo"
    if args.dry_run:
        for scale in ("small", "medium", "large"):
            workspace = build_workspace(entity, project, scale, ["example"])
            workspace._to_model()  # validate serialization, including group filters
            print(scale, "12 panels validated")
        return
    api = wandb.Api()
    index = args.root / "workspaces.json"
    urls = json.loads(index.read_text()) if index.exists() else {}
    for scale in ("small", "medium", "large"):
        runs = list(api.runs(f"{entity}/{project}", filters={"group": f"window-scale-{scale}"}))
        baselines = [
            r
            for r in runs
            if r.config.get("method") in {"random", "local", "forward", "queue-adaptive"}
        ]
        if len(baselines) != 5:
            raise ValueError(f"Expected five completed baselines for {scale}, got {len(baselines)}")
        for r in baselines:
            if r.state != "finished":
                raise ValueError(f"Baseline is not finished: {r.id}")
            if r.config.get("reference_only"):
                continue
            values = {"eval_stochastic/" + k: r.summary[k] for k in METRICS}
            with wandb.init(
                entity=entity, project=project, id=r.id, resume="must", dir=str(args.root)
            ) as run:
                run.config.update(
                    {
                        "reference_only": True,
                        "visualization_note": "Two endpoints of one fixed evaluation mean; not a training trajectory.",
                    },
                    allow_val_change=True,
                )
                run.define_metric("env_steps")
                run.define_metric("eval_stochastic/*", step_metric="env_steps")
                for step in (0, 262144):
                    run.log({"env_steps": step, **values})
        workspace = build_workspace(entity, project, scale, [r.id for r in baselines])
        if scale in urls:
            existing = ws.Workspace.from_url(urls[scale])
            existing.sections = workspace.sections
            existing.runset_settings = workspace.runset_settings
            workspace = existing
        workspace.save()
        urls[scale] = workspace.url
        index.write_text(json.dumps(urls, indent=2))
        loaded = ws.Workspace.from_url(workspace.url)
        assert len(loaded.sections[0].panels) == 8
        print(scale, workspace.url, flush=True)


if __name__ == "__main__":
    main()
