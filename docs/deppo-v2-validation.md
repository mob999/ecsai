# DEPPO v2 本地验收（2026-09-20）

环境：macOS ARM，CPU；生产依赖继续使用仓库 uv.lock。

- `python -m pytest -q`：180 passed，1 skipped（CUDA），19.78s。
- Ruff 检查通过；新增活动传输变速、容量分段积分、独立重复回源/取消测试。
- CPU smoke 验证 PPO 更新、GRU 梯度、actor 参数隔离、共享 critic、保存恢复、log probability 重算、KL 停止。
- 故障注入：Adam 更新产生 NaN 时立即停止，健康 `last.pt` 保持可读且参数有限；不做逐更新回滚。
- `scripts/run_deppo_v2.py --smoke` 完整走通固定比例搜索、两方法各4组搜索、两方法各3种子训练、独立测试、配对区间和1/2/4 worker benchmark。

Smoke 仅每次最终训练64环境步、每个方法3个测试episode，不能用于收敛或优劣结论：

| 方法 | 测试成功率均值 | 训练种子标准差 |
|---|---:|---:|
| DEPPO | 66.48% | 20.46% |
| MAPPO-no-context | 74.39% | 15.90% |
| Random | 98.48% | — |
| Local | 91.56% | — |
| Forward | 90.92% | — |
| Queue-adaptive | 39.66% | — |
| Tuned-fixed | 89.31% | — |

配置选择只使用验证工作负载；Tuned-fixed 在极小验证集的选择并不保证测试优于 Random。全部方法通过工作负载一致性检查。

另以 small 默认真实负载、4 workers、整批512、5轮更新运行2048环境步：4批训练无非有限数，平均步奖励约 -0.526/-0.526/-0.464/-0.462，各actor批次平均KL约2e-5–7e-5，每批完成5次策略更新。采样145–180环境步/s。两次固定验证episode成功率15.52%，仅是短跑结果，尚未收敛。

本地原始文件：`outputs/deppo-v2-smoke/`、`outputs/deppo-v2-stability/`。这些运行文件不提交仓库。使用 `python scripts/plot_deppo_v2.py <训练目录>` 从 metrics.csv 生成 reward、验证成功率、带宽比例、KL 的 `curves.svg`。正式实验必须另建目录并从头训练，不能继承此小预算配置。
