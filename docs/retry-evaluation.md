# 包含真实重传的冻结策略评估

用户确认规则（2026-09-21）：首次失败后最多重试 2 次，失败后等待 0.1 s，回到原始调度集群，同一内容和接收方；每次尝试独立限时 1 s。单次最多转发一次，重试时重新允许一次转发。重传重新竞争现有链路、队列和缓存，完整下载，不保留未完成的部分对象；已经完整缓存的对象可命中。回源合并等原场景配置不变。

原始 Poisson 请求在原 horizon 停止；随后停止产生新用户请求，但继续执行策略和重试，排空上界为 horizon + 3×deadline + 2×retry_delay。原始到达与重试同刻时原始到达先接纳；周期边界上的重试由后续周期接纳。精确截止时完成仍成功。重试默认关闭，不修改已有策略、reward 或训练。

一次尝试和一个用户请求不能混作分母：

- 原 `arrived/completed/timed_out/rejected/mean_latency_s` 继续表示尝试层，后者只包含成功尝试本身的时间。
- `logical_success_rate`：最终成功的原始请求数 / 原始请求数。
- `mean_success_e2e_s`：最终成功请求从首次到达到成功的总时间，含先前失败和重试等待。
- `mean_failed_elapsed_s`：重试耗尽仍失败请求从首次到达到最终失败的时间。
- `mean_resolution_time_s`：全部原始请求从首次到达到成功或最终失败的平均耗时。这不是“所有请求最终交付延迟”，必须与最终成功率一起看；快速拒绝会缩短它。
- `retry_attempts/mean_attempts`：实际额外尝试数与每原始请求平均尝试次数。
- `retry_outcomes`：每原始请求的逐次尝试记录，含开始、结束、结果、原始集群等；逐条验证总时间=各尝试耗时之和+重试等待。

公平性：冻结原 checkpoint，不重新训练；原三规模使用最终 checkpoint 和原 10 个验证种子，迁移实验使用原 best-stochastic 和原 30 个测试种子。两套结果不可跨表直接比较。所有规则与学习算法在排空期间继续按原周期控制，Random/Queue-adaptive 仍使用真实随机转发。仅发生重传后诱发的内生流量改变，原始工作负载完全相同。

运行：

```sh
.venv/bin/python scripts/evaluate_retries.py --scale-root /path/to/scale-matrix-v1 \
  --output outputs/retry-scale-v1 --workers 4
.venv/bin/python scripts/evaluate_retries.py --transfer-root /path/to/bc-transfer-v1 \
  --output outputs/retry-transfer-v1 --workers 4
```

每个完成案例保存 `spec.json`、完整 `evaluation.json`、W&B offline；根目录 `comparison.json` 逐案例更新，全部核对同一原始请求、尝试计数与排空后才生成 `complete.json`。可用相同参数续跑，完成案例跳过。日志包含完整逐请求数据，文件体积会大于旧指标文件。

本地验证：完整测试 200 passed / 2 skipped；验证部分交付后取消再重传、缓存命中、重试耗尽、溢出重试与边界顺序；迁移四分支和 DD 原 checkpoint 的 smoke 重评通过。
