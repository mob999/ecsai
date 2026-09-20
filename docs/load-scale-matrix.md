# 周期调度：三个规模场景

所有方法使用 window 调度、共享带宽、冷缓存、串行 FIFO 传输池、business reward，验证请求种子固定为 1000000000–1000000009。按用户最新要求，本轮只验证三个规模场景，暂不运行独立负载扫描：

| 场景 | 集群 | 缓存 | 请求/秒 | 理论交付负载比 |
|---|---:|---:|---:|---:|
| scale-small | 3 | 10 | 300 | 0.75 |
| scale-medium | 5 | 20 | 600 | 0.75 |
| scale-large | 7 | 30 | 900 | 0.75 |

规模组保持每缓存节点 30 请求/秒和平均容量相同；缓存、链路数量随规模增长。内容目录与 Zipf 不变，因此规模实验包含缓存数量增加及请求分散的效应。负载比是交付需求/总容量，不包含额外回源；实际目录采样后的期望负载写入各 episode 的 workload。

每场景五个规则：Random、Always-local、Always-forward、Forward-r0.6、Queue-adaptive；后者沿用逐请求随机转发，仅自适应带宽。0.6 是预先指定的参考值，不称为该场景的 Tuned-fixed。

每场景重新训练 MAPPO-no-context 和 DD-adapted，训练种子 0，各 262144 环境步。网络 256×256、学习率 3e-4、初始标准差 0.3、归一化 advantage、每批 1024 步、5 轮更新。默认两个独立训练任务并行，各 8 采样 worker，评估各 4 worker。固定 8192 步/10 episodes 验证，同时保留随机与确定性执行。该轮是单训练种子的场景对比，不作为多种子最终统计结论。

运行：

```sh
.venv/bin/python scripts/run_load_matrix.py --smoke --output outputs/scale-matrix-smoke
.venv/bin/python scripts/run_load_matrix.py --device cuda --wandb-mode online --output outputs/scale-matrix-v1
```

运行入口先完成所有规则基线，再按场景同时训练 DD/MAPPO。W&B 名称和分组含场景，防止跨场景混比；每项保存 scenario/job/config/evaluation、CSV、checkpoint 和 console.log。失败时保留训练现有 last.pt；重启同一命令会跳过完成项并恢复未完成训练。进程退出成功且最终验证文件存在才标完成。

全套结束时校验每场景的方法使用相同配置、种子、请求数量与 workload 元信息；验证排空后请求计数闭合，再生成 comparison.json/csv 和 suite complete.json。必须另行检查实际训练进度及最终 checkpoint，不能仅凭启动记录宣称实验完成。

本地验证：全仓 194 passed、1 skipped（CUDA）；三规模缩小工作负载的 CPU smoke 完成全部 21 项（15 规则评估、6 训练），6 个训练 last.pt 均已保存。汇总校验配置、工作负载、排空计数通过；再次运行相同入口跳过完成项并成功重新汇总。正式 3/10、5/20、7/30 场景 build_run 均通过。Ruff 与 diff check 通过。
