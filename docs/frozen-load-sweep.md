# 冻结 checkpoint 的请求负载扫描

沿用 590dee6 的真实重传规则：最多 2 次重试、间隔 0.1 s、每次尝试限时 1 s。原有模型不再训练，不改变资源预算或按结果选择 checkpoint。

预先固定负载比为 0.25、0.50、0.75、1.00、1.25，三种规模分别使用各自原最终 MAPPO、DD checkpoint。每个条件使用原 10 个验证种子，所有方法共享原始请求。比较 MAPPO、DD、Random、Always-local、Always-forward、Forward-r0.6、Queue-adaptive，共 105 个条件、1,050 个 episode；0.75 档复用已经完成的 retry-scale-v1，其余 84 个条件新跑。这是探索性负载扫描，不是独立最终测试集或跨训练种子结论。

固定硬件容量，只调整请求到达率。校准器容量正比于 request_rate/delivery_load，因此同时按目标负载缩放二者，使节点总容量保持原值。脚本逐节点校验带宽、归属不变。小/中/大原到达率为 300/600/900 请求每秒、原负载比为 0.75，对应扫描到达率分别为 100–500、200–1000、300–1500 请求每秒。负载比仅计理论原始交付流量，额外回源与实际重传会继续增加资源需求。

主要指标是最终逻辑请求成功率；同时报告成功请求总延迟、耗尽重试的失败耗时、全部请求结束耗时和平均尝试次数。不能把快速拒绝引起的短结束耗时解释为性能改善。所有负载均报告，不只保留学习策略占优的条件。

运行（每个规模独立进程，建议每进程 4 worker）：

```sh
.venv/bin/python scripts/evaluate_load_sweep.py \
  --root /root/ecsai/outputs/scale-matrix-v1 --size small \
  --output outputs/load-sweep-v1/small --workers 4
```

逐条件保存 spec、checkpoint SHA256、紧凑 episode 指标、gzip 逐请求重传轨迹及 W&B offline 日志。相同配置可续跑，已完成条件跳过；不同配置复用输出目录时报错。根目录 comparison.json 随完成更新，全部条件完成写 complete.json。
